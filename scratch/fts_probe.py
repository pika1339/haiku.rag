"""Probe how haiku.rag's FTS path handles Chinese text.

Zero model cost: chunks are imported with fake embeddings and the embedder is
patched to raise, so any embedder call fails loudly instead of hitting a server.
"""

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, set_config
from haiku.rag.embeddings import EmbedderWrapper
from haiku.rag.store.models.chunk import Chunk

VECTOR_DIM = 2560


class EmbedderCalled(AssertionError):
    pass


def _boom(*args, **kwargs):
    raise EmbedderCalled("the embedder was called; this probe must stay free")


def patch_embedder():
    EmbedderWrapper.embed_query = _boom
    EmbedderWrapper.embed_documents = _boom
    EmbedderWrapper.embed_image = _boom


# --- corpus ---------------------------------------------------------------

ZH_SHORT = "混合检索效果好。"
ZH_LONG_RUN = (
    "本研究采用混合检索方法并在中文科技文献数据集上验证了其有效性"
)  # 29 chars, no internal punctuation
ZH_PARAS = [
    "检索增强生成（Retrieval-Augmented Generation, RAG）通过在生成阶段引入外部知识库，显著缓解了大语言模型的幻觉问题。",
    "本文提出一种面向中文科技文献的混合检索方法，将稀疏检索与稠密检索的结果进行融合。",
    "实验在中文科技文献数据集上进行。我们采用平均倒数排名（MRR）和召回率作为评价指标。",
    "实验结果表明，混合检索的召回率比单一稠密检索提高了百分之十二。",
    "然而，稀疏检索依赖分词质量。中文文本没有天然的词边界，若分词器未能正确切分，倒排索引的匹配能力会显著下降。",
]
EN_PARAS = [
    "Retrieval-augmented generation mitigates hallucination in large language models.",
    "We propose a hybrid retrieval method for scientific literature.",
    "Experiments show that the recall of hybrid retrieval exceeds dense retrieval by twelve percent.",
]

ZH_QUERIES = [
    "混合检索",
    "召回率",
    "分词",
    "中文科技文献",
    "稀疏检索",
    "分词器未能正确切分",  # sentence-level query
]
EN_QUERIES = ["hybrid retrieval", "recall", "hallucination"]


def build_doc(name: str, paragraphs: list[str]) -> DoclingDocument:
    doc = DoclingDocument(name=name)
    for p in paragraphs:
        doc.add_text(label=DocItemLabel.TEXT, text=p)
    return doc


def make_chunks(paragraphs: list[str]) -> list[Chunk]:
    return [
        Chunk(content=p, embedding=[0.0] * VECTOR_DIM, order=i)
        for i, p in enumerate(paragraphs)
    ]


# --- probe ----------------------------------------------------------------


async def fts(repo, query: str, limit: int = 5):
    return await repo.search(query=query, limit=limit, search_type="fts")


def report(tag: str, query: str, hits):
    if not hits:
        print(f"  [{tag}] {query!r:<34} -> NO HITS")
        return
    top = hits[0]
    snippet = top[0].content[:40].replace("\n", " ")
    print(f"  [{tag}] {query!r:<34} -> {len(hits)} hit(s), top score {top[1]:.4f} | {snippet}...")


async def main() -> int:
    patch_embedder()

    cfg = AppConfig()
    cfg.storage.auto_vacuum = False
    set_config(cfg)

    root = Path(tempfile.mkdtemp(prefix="lit-agent-fts-"))
    db = root / "probe.lancedb"
    print(f"db: {db}\n")

    try:
        async with HaikuRAG(db, config=cfg, create=True) as rag:
            await rag.import_document(
                build_doc("zh", ZH_PARAS + [ZH_SHORT, ZH_LONG_RUN]),
                make_chunks(ZH_PARAS + [ZH_SHORT, ZH_LONG_RUN]),
                uri="zh://paper-1",
                title="中文测试文档",
            )
            await rag.import_document(
                build_doc("en", EN_PARAS),
                make_chunks(EN_PARAS),
                uri="en://paper-1",
                title="English test doc",
            )
            print("imported with the embedder patched to raise: OK\n")

            repo = rag.chunk_repository
            table = rag.store.chunks_table
            rows = await table.count_rows()
            print(f"chunks rows: {rows}")
            indices = await table.list_indices()
            for i in indices:
                print(f"  index: {i.name} type={i.index_type} cols={i.columns}")

            print("\n--- English control (default 'simple' tokenizer) ---")
            for q in EN_QUERIES:
                report("en", q, await fts(repo, q))

            print("\n--- Chinese (default 'simple' tokenizer) ---")
            for q in ZH_QUERIES + [ZH_SHORT, ZH_LONG_RUN]:
                report("zh", q, await fts(repo, q))

            print("\n--- exact-run queries against the two crafted paragraphs ---")
            for q in [ZH_SHORT, ZH_LONG_RUN]:
                report("run", q, await fts(repo, q))

            print("\n--- rebuild the FTS index with base_tokenizer='icu' ---")
            from lancedb.index import FTS

            try:
                await table.create_index(
                    "content_fts",
                    config=FTS(
                        with_position=True,
                        remove_stop_words=False,
                        base_tokenizer="icu",
                    ),
                    replace=True,
                )
                print("icu index built")
            except Exception as exc:  # noqa: BLE001
                print(f"icu index FAILED: {type(exc).__name__}: {exc}")
            else:
                repo._fts_coverage_checked = False
                print("\n--- Chinese under icu tokenizer ---")
                for q in ZH_QUERIES + [ZH_SHORT, ZH_LONG_RUN]:
                    report("icu", q, await fts(repo, q))
                print("\n--- English under icu tokenizer (control) ---")
                for q in EN_QUERIES:
                    report("icu", q, await fts(repo, q))

            return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
