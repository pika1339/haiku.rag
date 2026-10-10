"""Chinese FTS diagnosis on the real corpus (phase 2).

Phase 1 established the stored text is clean, so any failure measured here is a
retrieval failure, not a parse failure.

Method: for every candidate term, first confirm it appears VERBATIM in some
stored chunk (that is the ground truth), then ask the full-text index for it and
measure whether those chunks come back. A term that is provably in the corpus
and cannot be retrieved is the defect, stated without interpretation.

Runs the whole measure twice: once on the index haiku.rag declares, once on the
same index built with base_tokenizer='icu'. No embedder is called either way.
"""

import asyncio
import re
import shutil
import sys
import tempfile
from pathlib import Path

from lancedb.index import FTS

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, set_config
from haiku.rag.embeddings import EmbedderWrapper

LIMIT = 10
CJK = re.compile(r"[一-鿿]")

EXACT_TERMS = [
    "变分模态分解",
    "光伏功率预测",
    "相似日选取",
    "量子长短期记忆网络",
    "白鲸优化算法",
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
    # a long clause, past the 13-character cliff
    "针对光伏功率预测中极端天气样本不足",
]

EN_CONTROL = ["VMD", "KELM", "photovoltaic", "forecasting"]

NL_QUESTIONS = [
    "光伏功率预测中极端天气样本不足的问题怎么解决",
    "哪篇论文用了量子长短期记忆网络",
    "几篇论文分别用了什么评价指标",
    "VMD 在模型里起什么作用",
]


def stub_embedder() -> None:
    async def fake_documents(self, texts):
        return [[0.0] * self._vector_dim for _ in texts]

    async def fake_query(self, text):
        return [[0.0] * self._vector_dim]

    EmbedderWrapper._embed_documents = fake_documents
    EmbedderWrapper._embed_query = fake_query


def cliffs(contents: list[str]) -> tuple[int, int]:
    """(CJK chars inside runs over 40 UTF-8 bytes, total CJK chars)."""
    over = total = 0
    for text in contents:
        for run in re.findall(r"[一-鿿]+", text):
            total += len(run)
            if len(run.encode("utf-8")) > 40:
                over += len(run)
    return over, total


async def main(pdf_dir: str) -> int:
    stub_embedder()

    cfg = AppConfig()
    cfg.storage.auto_vacuum = False
    set_config(cfg)

    pdfs = sorted(Path(pdf_dir).glob("*.pdf"))
    root = Path(tempfile.mkdtemp(prefix="lit-agent-zh2-"))
    db = root / "zh.lancedb"
    print(f"db: {db}\ningesting {len(pdfs)} PDF(s)...")

    try:
        async with HaikuRAG(db, config=cfg, create=True) as rag:
            for pdf in pdfs:
                doc = await rag.create_document_from_source(pdf)
                print(f"  {pdf.name[:40]:<42} id={doc.id}")

            rows = await (
                rag.store.chunks_table.query()
                .select(["id", "content"])
                .to_list()
            )
            contents = [r["content"] for r in rows]
            id_by_index = {i: r["id"] for i, r in enumerate(rows)}
            print(f"\nstored chunks: {len(contents)}")

            # ---- report measures the corpus and the index ----------------------
            over, total = cliffs(contents)
            print(
                f"13-char cliff on the real corpus: {over}/{total} CJK chars "
                f"= {over / total:.1%} sit in runs over 40 UTF-8 bytes"
            )

            async def run_block(label: str) -> None:
                repo = rag.chunk_repository
                print(f"\n{'=' * 74}\n{label}\n{'=' * 74}")
                print(f"{'term':<36} {'truth':>6} {'hits':>5} {'recall@10':>10} {'best rank':>9}")
                for group, terms in (("ZH", EXACT_TERMS), ("EN", EN_CONTROL)):
                    for term in terms:
                        truth = [i for i, c in enumerate(contents) if term in c]
                        if not truth:
                            print(f"{term[:34]:<36} {'—':>6}   (absent from corpus)")
                            continue
                        hits = await repo.search(
                            query=term, limit=LIMIT, search_type="fts"
                        )
                        order = [h[0].id for h in hits]
                        truth_ids = {id_by_index[i] for i in truth}
                        found = sum(1 for cid in truth_ids if cid in order)
                        best = next(
                            (r + 1 for r, cid in enumerate(order) if cid in truth_ids),
                            None,
                        )
                        print(
                            f"[{group}] {term[:30]:<30} {len(truth):>6} {len(hits):>5} "
                            f"{found / len(truth):>9.0%} {str(best):>9}"
                        )

                print("\nnatural-language questions (what a user would ask):")
                for q in NL_QUESTIONS:
                    hits = await repo.search(query=q, limit=LIMIT, search_type="fts")
                    top = hits[0][0].content[:34].replace("\n", " ") if hits else "—"
                    print(f"  {q[:34]:<36} -> {len(hits):>2} hits | top: {top}")

            await run_block("A. INDEX AS haiku.rag DECLARES IT (base_tokenizer='simple')")

            print(f"\n{'=' * 74}\nrebuilding the FTS index with base_tokenizer='icu'\n{'=' * 74}")
            await rag.store.chunks_table.create_index(
                "content_fts",
                config=FTS(with_position=True, remove_stop_words=False, base_tokenizer="icu"),
                replace=True,
            )
            rag.chunk_repository._fts_coverage_checked = False
            await run_block("B. SAME ROWS, INDEX REBUILT WITH base_tokenizer='icu'")
            return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "papers")))
