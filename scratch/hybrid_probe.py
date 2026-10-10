"""Does the hybrid path the capability actually uses recover from a vector-side miss?

Each chunk gets a one-hot vector, and each query vector is set to a *wrong*
chunk's one-hot. So the vector side always ranks a decoy first and the true
match last; only the full-text side can rescue it. Compares Chinese with English.

Calls ChunkRepository.search directly with a pre-computed query_vector, so no
embedder is invoked (the capability path is identical apart from the embedding).
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

# (label, text, one-hot slot)
ROWS = [
    ("zh_true", "本文验证混合检索对中文召回率的提升", 0),
    ("zh_decoy", "光伏功率预测模型研究", 1),
    ("en_true", "this paper validates that hybrid retrieval improves recall", 2),
    ("en_decoy", "photovoltaic power forecasting model", 3),
]


def one_hot(slot: int) -> list[float]:
    v = [0.0] * VECTOR_DIM
    v[slot] = 1.0
    return v


def _boom(*args, **kwargs):
    raise AssertionError("embedder was called")


async def run(repo, query: str, search_type: str, query_vector) -> list[str]:
    hits = await repo.search(
        query=query,
        limit=4,
        search_type=search_type,
        query_vector=query_vector,
    )
    return [c.content[:26] for c, _ in hits]


async def main() -> int:
    EmbedderWrapper.embed_query = _boom
    EmbedderWrapper.embed_documents = _boom

    cfg = AppConfig()
    cfg.storage.auto_vacuum = False
    set_config(cfg)

    root = Path(tempfile.mkdtemp(prefix="lit-agent-hybrid-"))
    db = root / "h.lancedb"
    try:
        doc = DoclingDocument(name="h")
        for _, text, _ in ROWS:
            doc.add_text(label=DocItemLabel.TEXT, text=text)

        async with HaikuRAG(db, config=cfg, create=True) as rag:
            await rag.import_document(
                doc,
                [
                    Chunk(content=text, embedding=one_hot(slot), order=i)
                    for i, (_, text, slot) in enumerate(ROWS)
                ],
                uri="mix://1",
                title="mix",
            )
            repo = rag.chunk_repository
            zh_decoy_vec = one_hot(1)  # points at the *wrong* Chinese chunk
            en_decoy_vec = one_hot(3)  # points at the *wrong* English chunk

            cases = [
                ("ZH  query '混合检索'", "混合检索", zh_decoy_vec),
                ("EN  query 'hybrid retrieval'", "hybrid retrieval", en_decoy_vec),
            ]
            for title, query, vec in cases:
                print(f"\n=== {title} ===")
                for st in ("vector", "fts", "hybrid"):
                    qv = vec if st != "fts" else None
                    try:
                        order = await run(repo, query, st, qv)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  {st:<7} ERROR {type(exc).__name__}: {exc}")
                        continue
                    print(f"  {st:<7} -> {order}")
            return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
