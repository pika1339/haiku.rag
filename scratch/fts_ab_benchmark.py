"""A/B the FTS base_tokenizer on a Chinese corpus and an English control.

Both corpora are ingested into one database with the REAL pipeline (docling-local
conversion and chunking); only the embedding is a zero-vector stub, since an FTS
comparison reads no vector.

Ground truth is verbatim presence: a chunk is relevant to a term when the term
appears in it. That measures retrievability, not semantic relevance, which is
exactly what a tokenizer change can move. The metric formulas are the base's own
(`average_precision` is imported; recall and nDCG mirror
`evaluations/evaluators/retrieval.py`), applied at chunk granularity because three
Chinese documents cannot support document-level retrieval metrics.

Two arms, same rows: the index as haiku.rag declares it, then the same index
rebuilt with base_tokenizer='icu'.
"""

import asyncio
import json
import math
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from lancedb.index import FTS

from evaluations.evaluators.citation import average_precision
from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, set_config
from haiku.rag.embeddings import EmbedderWrapper

VECTOR_DIM = 2560
KS = (5, 10)
FETCH = 20
CJK = re.compile(r"[一-鿿]")
EN_WORD = re.compile(r"[a-zA-Z][a-zA-Z'-]{4,}")

ZH_MANUAL = [
    "变分模态分解",
    "光伏功率预测",
    "相似日选取",
    "量子长短期记忆网络",
    "核极限学习机",
    "动态时间规整",
    "预测精度",
    "评价指标",
    "平均绝对百分比误差",
    "相关系数",
    "聚类",
    "光伏电站",
    "归一化",
    "测试集",
    "针对光伏功率预测中极端天气样本不足",
]

ZH_NL = [
    "光伏功率预测中极端天气样本不足的问题怎么解决",
    "哪篇论文用了量子长短期记忆网络",
    "几篇论文分别用了什么评价指标",
    "VMD 在模型里起什么作用",
    "相似日是怎么选取的",
    "KELM 的参数怎么优化",
    "各模型在多云天气下的表现差异",
    "论文用了哪些数据预处理方法",
]


def stub_embedder() -> None:
    async def fake_documents(self, texts):
        return [[0.0] * self._vector_dim for _ in texts]

    async def fake_query(self, text):
        return [[0.0] * self._vector_dim]

    EmbedderWrapper._embed_documents = fake_documents
    EmbedderWrapper._embed_query = fake_query


# --- metrics, mirroring evaluations/evaluators/retrieval.py at chunk granularity


def recall_at(ranked: list[str], truth: set[str], k: int) -> float:
    if not truth:
        return 0.0
    return sum(1 for cid in ranked[:k] if cid in truth) / len(truth)


def ndcg_at(ranked: list[str], truth: set[str], k: int) -> float:
    if not truth:
        return 0.0
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, cid in enumerate(ranked[:k], start=1)
        if cid in truth
    )
    ideal = sum(
        1 / math.log2(rank + 1) for rank in range(1, min(len(truth), k) + 1)
    )
    return dcg / ideal


# --- query sets


def zh_auto(contents: list[str], want: int = 40) -> list[str]:
    seen: dict[str, int] = defaultdict(int)
    for text in contents:
        grams = set()
        for run in re.findall(r"[一-鿿]+", text):
            for n in (4, 5):
                for i in range(len(run) - n + 1):
                    grams.add(run[i : i + n])
        for g in grams:
            seen[g] += 1
    pool = sorted(g for g, c in seen.items() if 1 <= c <= 8)
    if len(pool) <= want:
        return pool
    step = len(pool) / want
    return [pool[int(i * step)] for i in range(want)]


def en_auto(contents: list[str], want: int = 60) -> list[str]:
    seen: dict[str, int] = defaultdict(int)
    for text in contents:
        words = EN_WORD.findall(text.lower())
        grams = set(words)
        grams |= {f"{a} {b}" for a, b in zip(words, words[1:])}
        for g in grams:
            seen[g] += 1
    pool = sorted(g for g, c in seen.items() if 1 <= c <= 8)
    if len(pool) <= want:
        return pool
    step = len(pool) / want
    return [pool[int(i * step)] for i in range(want)]


# --- measurement


async def arm(repo, queries: list[str], contents: list[str], ids: list[str]):
    """Per-query metrics for one index configuration."""
    per_query = []
    empty = 0
    for q in queries:
        truth = {ids[i] for i, c in enumerate(contents) if q in c}
        hits = await repo.search(query=q, limit=FETCH, search_type="fts")
        ranked = [h[0].id for h in hits]
        if not ranked:
            empty += 1
        if not truth:
            per_query.append(None)
            continue
        row = {"map": average_precision(ranked, truth), "empty": not ranked}
        for k in KS:
            row[f"recall@{k}"] = recall_at(ranked, truth, k)
            row[f"ndcg@{k}"] = ndcg_at(ranked, truth, k)
        per_query.append(row)
    scored = [r for r in per_query if r]
    if not scored:
        return None
    agg = {
        "n": len(scored),
        "zero_hit_rate": empty / len(queries),
        "map": sum(r["map"] for r in scored) / len(scored),
    }
    for k in KS:
        agg[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in scored) / len(scored)
        agg[f"ndcg@{k}"] = sum(r[f"ndcg@{k}"] for r in scored) / len(scored)
    return agg


METRIC_ROWS = ["map", "recall@5", "recall@10", "ndcg@5", "ndcg@10", "zero_hit_rate"]


def show(block: str, a: dict | None, b: dict | None) -> None:
    print(f"\n### {block}")
    if a is None or b is None:
        print("  (no scorable queries)")
        return
    print(f"  queries scored: {a['n']}")
    print(f"  {'metric':<14} {'simple':>9} {'icu':>9} {'delta':>9}")
    for m in METRIC_ROWS:
        av, bv = a[m], b[m]
        delta = bv - av
        flag = ""
        if m == "zero_hit_rate":
            flag = "  <-- lower is better"
        elif delta > 0.005:
            flag = "  <-- BETTER"
        elif delta < -0.005:
            flag = "  <-- WORSE"
        print(f"  {m:<14} {av:>9.4f} {bv:>9.4f} {delta:>+9.4f}{flag}")


async def nl_hit_rate(repo, questions: list[str]) -> tuple[float, float]:
    hit = 0
    for q in questions:
        if await repo.search(query=q, limit=FETCH, search_type="fts"):
            hit += 1
    return hit / len(questions), hit


async def main(pdf_dir: str, en_corpus: str) -> int:
    stub_embedder()

    cfg = AppConfig()
    cfg.storage.auto_vacuum = False
    set_config(cfg)

    root = Path(tempfile.mkdtemp(prefix="lit-agent-ab-"))
    db = root / "ab.lancedb"
    pdfs = sorted(Path(pdf_dir).glob("*.pdf"))
    en_docs = json.loads(Path(en_corpus).read_text(encoding="utf-8"))
    print(f"db: {db}\nzh: {len(pdfs)} PDF(s)   en: {len(en_docs)} document(s)")

    # Both corpora enter through create_document_from_source so the two arms and
    # the two corpora share one ingestion path.
    en_dir = root / "en"
    en_dir.mkdir()
    en_paths = []
    for item in en_docs:
        path = en_dir / f"{item['document_id']}.md"
        path.write_text(item["document_extracted"], encoding="utf-8")
        en_paths.append(path)

    try:
        async with HaikuRAG(db, config=cfg, create=True) as rag:
            for pdf in pdfs:
                doc = await rag.create_document_from_source(pdf)
                print(f"  zh  {pdf.name[:38]:<40} {doc.id}")
            for i, path in enumerate(en_paths):
                await rag.create_document_from_source(path)
                if i % 10 == 0:
                    print(f"  en  {i + 1}/{len(en_paths)}")

            rows = await (
                rag.store.chunks_table.query().select(["id", "content", "document_id"]).to_list()
            )
            meta = await (
                rag.store.document_meta_table.query().select(["id", "uri"]).to_list()
            )
            uri_by_doc = {m["id"]: m["uri"] for m in meta}
            ids = [r["id"] for r in rows]
            contents = [r["content"] for r in rows]
            from_pdf = [
                str(uri_by_doc.get(r["document_id"], "")).lower().endswith(".pdf")
                for r in rows
            ]
            cjk_len = [len(CJK.findall(c)) for c in contents]
            groups = {
                "zh_pdf": [i for i in range(len(rows)) if from_pdf[i] and cjk_len[i] > 20],
                "en_pdf": [i for i in range(len(rows)) if from_pdf[i] and cjk_len[i] <= 5],
                "en_docs": [i for i in range(len(rows)) if not from_pdf[i]],
            }
            print(f"\nchunks total={len(contents)}")
            for name, idxs in groups.items():
                lens = [len(contents[i]) for i in idxs]
                med = sorted(lens)[len(lens) // 2] if lens else 0
                print(
                    f"  {name:<8} n={len(idxs):>4}  chars: median={med} max={max(lens) if lens else 0}"
                )

            zh_q = ZH_MANUAL + zh_auto([contents[i] for i in groups["zh_pdf"]])
            en_pdf_q = en_auto([contents[i] for i in groups["en_pdf"]], want=30)
            en_docs_q = en_auto([contents[i] for i in groups["en_docs"]], want=40)
            en_nl = [d["question"] for d in en_docs if d.get("question")][:25]
            print(
                f"queries: zh manual={len(ZH_MANUAL)} zh auto={len(zh_q) - len(ZH_MANUAL)} "
                f"en_pdf={len(en_pdf_q)} en_docs={len(en_docs_q)} en nl={len(en_nl)}"
            )

            results = {}
            for arm_name, tokenizer in (("simple", None), ("icu", "icu")):
                if tokenizer:
                    await rag.store.chunks_table.create_index(
                        "content_fts",
                        config=FTS(
                            with_position=True,
                            remove_stop_words=False,
                            base_tokenizer=tokenizer,
                        ),
                        replace=True,
                        name="content_fts_idx",
                    )
                    rag.chunk_repository._fts_coverage_checked = False
                repo = rag.chunk_repository
                results[arm_name] = {
                    "zh_manual": await arm(repo, ZH_MANUAL, contents, ids),
                    "zh_auto": await arm(repo, zh_q[len(ZH_MANUAL):], contents, ids),
                    "en_pdf": await arm(repo, en_pdf_q, contents, ids),
                    "en_docs": await arm(repo, en_docs_q, contents, ids),
                    "zh_nl": await nl_hit_rate(repo, ZH_NL),
                    "en_nl": await nl_hit_rate(repo, en_nl),
                }

            print("\n" + "=" * 74)
            print("RESULTS — 改前 (as declared) vs 改后 (base_tokenizer='icu')")
            print("=" * 74)
            for block in ("zh_manual", "zh_auto", "en_pdf", "en_docs"):
                show(block, results["simple"][block], results["icu"][block])

            print("\n### natural-language questions (hit rate = share returning >=1 result)")
            for lang in ("zh_nl", "en_nl"):
                a, an = results["simple"][lang]
                b, bn = results["icu"][lang]
                print(
                    f"  {lang:<8} simple {a:.0%} ({an})   icu {b:.0%} ({bn})   "
                    f"delta {(b - a) * 100:+.0f}pp"
                )
            return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(
        asyncio.run(
            main(
                sys.argv[1] if len(sys.argv) > 1 else "papers",
                sys.argv[2] if len(sys.argv) > 2 else "tests/data/qa_corpus.json",
            )
        )
    )
