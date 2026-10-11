"""The three retrieval arms over the real corpus with real bge-m3 vectors.

Answers what the FTS-only measurement could not: with the vector leg in play,
how much of Chinese retrieval was already covered, and what does fixing the
tokenizer add on top.

Two ground truths, because one of them is unfair to one of the arms:

- literal: a chunk contains the query term verbatim. Exact for the FTS arm, and
  a lower bound for the vector arm, which is *supposed* to return related text
  that does not contain the term.
- topical: the documents whose text contains the distinguishing term of the
  question. Document-level, so all three arms are scored on the same thing. The
  term sets are printed, not assumed, so a wrong label is visible rather than
  silently in the average.

Query embeddings come from the configured embedder, so the vector and hybrid
arms call Ollama. The FTS arm calls nothing.
"""

import asyncio
import json
import math
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from evaluations.evaluators.citation import average_precision
from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, set_config

CJK = re.compile(r"[一-鿿]")
EN_WORD = re.compile(r"[a-zA-Z][a-zA-Z'-]{4,}")
KS = (5, 10)
FETCH = 20

# (natural-language question, the distinctive term that decides the answer)
TOPIC_QUERIES = [
    ("哪篇论文提出了基于量子长短期记忆网络的预测模型？", "量子长短期记忆"),
    ("哪篇论文使用了核极限学习机？", "极限学习机"),
    ("哪篇论文用动态时间规整做聚类？", "动态时间规整"),
    ("哪篇论文做了相似日选取？", "相似日"),
    ("哪篇论文用白鲸优化算法优化参数？", "白鲸"),
    ("哪篇论文用 Pearson 相关系数构建加权特征矩阵？", "Pearson"),
    ("哪篇论文研究的是分布式光伏？", "分布式光伏"),
    ("哪篇论文提出了 NAHA 算法？", "NAHA"),
    ("哪篇论文研究了边界效应和模态混叠？", "模态混叠"),
    ("哪篇论文用变分模态分解？", "变分模态分解"),
]

ZH_TERMS = [
    "变分模态分解", "光伏功率预测", "相似日选取", "量子长短期记忆网络",
    "核极限学习机", "动态时间规整", "预测精度", "评价指标", "相关系数", "光伏电站",
]


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


def recall_at(ranked: list[str], truth: set[str], k: int) -> float:
    if not truth:
        return 0.0
    return sum(1 for x in ranked[:k] if x in truth) / min(len(truth), k)


def ndcg_at(ranked: list[str], truth: set[str], k: int) -> float:
    if not truth:
        return 0.0
    dcg = sum(1 / math.log2(r + 1) for r, x in enumerate(ranked[:k], start=1) if x in truth)
    ideal = sum(1 / math.log2(r + 1) for r in range(1, min(len(truth), k) + 1))
    return dcg / ideal


METRICS = ["map", "recall@5", "recall@10", "ndcg@5", "ndcg@10", "zero_hit_rate"]


async def run_arm(repo, queries, truth_of, key_of):
    """truth_of(q) -> set of ids; key_of(chunk_or_id) -> the id space being scored."""
    agg = defaultdict(list)
    empty = 0
    for q in queries:
        truth = truth_of(q)
        hits = await repo.search(query=q, limit=FETCH)
        ranked = [key_of(h[0]) for h in hits]
        if not ranked:
            empty += 1
        if not truth:
            continue
        agg["map"].append(average_precision(ranked, truth))
        for k in KS:
            agg[f"recall@{k}"].append(recall_at(ranked, truth, k))
            agg[f"ndcg@{k}"].append(ndcg_at(ranked, truth, k))
    if not agg["map"]:
        return None
    out = {"n": len(agg["map"]), "zero_hit_rate": empty / len(queries)}
    for m in METRICS:
        if m != "zero_hit_rate":
            out[m] = sum(agg[m]) / len(agg[m])
    return out


class Armed:
    """A ChunkRepository with an arm name, since search() takes no arm label."""

    def __init__(self, repo, arm):
        self._repo = repo
        self._arm = arm

    async def search(self, **kw):
        return await self._repo.search(search_type=self._arm, **kw)


def table(title: str, per_arm: dict[str, dict | None]) -> None:
    print(f"\n### {title}")
    arms = [a for a in per_arm if per_arm[a]]
    if not arms:
        print("  (no scorable queries)")
        return
    print(f"  {'metric':<14} " + "".join(f"{a:>10}" for a in arms))
    for m in METRICS:
        row = "".join(f"{per_arm[a][m]:>10.4f}" for a in arms)
        print(f"  {m:<14}{row}")


async def main(pdf_dir: str) -> int:
    cfg = AppConfig()
    cfg.storage.auto_vacuum = False
    cfg.embeddings.model.provider = "ollama"
    cfg.embeddings.model.name = "bge-m3"
    cfg.embeddings.model.vector_dim = 1024
    # Ollama rejects a larger request for bge-m3 (400 at 300 inputs, 256 is the
    # last size that works); the default 512 never gets through.
    cfg.embeddings.batch_size = 128
    set_config(cfg)

    pdfs = sorted(Path(pdf_dir).glob("*.pdf"))
    root = Path(tempfile.mkdtemp(prefix="lit-agent-baseline-"))
    db = root / "base.lancedb"
    print(f"db: {db}")
    print(f"embedder: {cfg.embeddings.model.provider}/{cfg.embeddings.model.name} "
          f"dim={cfg.embeddings.model.vector_dim}")

    try:
        async with HaikuRAG(db, config=cfg, create=True) as rag:
            t0 = time.time()
            doc_ids = {}
            for pdf in pdfs:
                doc = await rag.create_document_from_source(pdf)
                doc_ids[doc.id] = pdf.stem
                print(f"  {pdf.stem[:44]:<46} {doc.id}")
            print(f"ingested in {time.time() - t0:.1f}s")

            rows = await (
                rag.store.chunks_table.query()
                .select(["id", "content", "document_id"]).to_list()
            )
            ids = [r["id"] for r in rows]
            contents = [r["content"] for r in rows]
            doc_of = dict(zip(ids, [r["document_id"] for r in rows]))
            n_chunks = len(contents)
            print(f"chunks: {n_chunks}  "
                  f"({n_chunks / len(pdfs):.1f}/doc)")
            print(f"embedding throughput: {n_chunks / (time.time() - t0):.1f} chunks/s "
                  "including parsing")

            doc_text = defaultdict(str)
            for out_doc, c in zip(doc_of.values(), contents):
                doc_text[out_doc] += c

            # --- topical ground truth, printed rather than assumed ------------
            print("\n=== topical ground truth (documents whose text holds the term) ===")
            topic_truth = {}
            for q, term in TOPIC_QUERIES:
                holders = [doc_ids[d] for d in doc_text if term in doc_text[d]]
                topic_truth[q] = {d for d in doc_text if term in doc_text[d]}
                flag = "  <-- not discriminating" if len(holders) > 1 else ""
                print(f"  {term:<16} -> {len(holders)} doc(s): "
                      f"{', '.join(h[:16] for h in holders)}{flag}")

            zh_chunks = [i for i, c in enumerate(contents) if len(CJK.findall(c)) > 20]
            zh_q = ZH_TERMS + zh_auto([contents[i] for i in zh_chunks])

            print(f"\nqueries: zh terms={len(ZH_TERMS)} zh auto={len(zh_q) - len(ZH_TERMS)} "
                  f"topical={len(TOPIC_QUERIES)}")

            per_topic = {}
            per_literal = {}
            for arm_name in ("vector", "fts", "hybrid"):
                armed = Armed(rag.chunk_repository, arm_name)
                per_topic[arm_name] = await run_arm(
                    armed, [q for q, _ in TOPIC_QUERIES],
                    lambda q: topic_truth[q], key_of=lambda c: c.document_id,
                )
                per_literal[arm_name] = await run_arm(
                    armed, zh_q, lambda q: {ids[i] for i, c in enumerate(contents) if q in c},
                    key_of=lambda c: c.id,
                )

            print("\n" + "=" * 70)
            print("RESULTS — three arms, real bge-m3 vectors")
            print("=" * 70)
            table("topical (document-level) — fair to all three arms", per_topic)
            table("literal (chunk-level) — exact for FTS, a lower bound for vector",
                  per_literal)
            return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "papers")))
