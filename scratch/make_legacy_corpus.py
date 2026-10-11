"""Build the "retrieval unfixed" corpus: same rows, index built the old way.

Copies the corpus and replaces only the chunks FTS index with the one haiku.rag
declared before the ICU change. Chunks, vectors and documents are the copy's, so
the two corpora differ in the index alone — a paired comparison of the retrieval
condition rather than of two ingests.
"""

import asyncio
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SRC = Path(".data/corpus.lancedb")
DST = Path(".data/corpus_legacy.lancedb")


async def main() -> int:
    from lancedb.index import FTS

    from haiku.rag.config import AppConfig, set_config
    from haiku.rag.store.engine import Store

    cfg = AppConfig()
    cfg.embeddings.model.name = "bge-m3"
    cfg.embeddings.model.vector_dim = 1024
    set_config(cfg)

    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)
    print(f"copied {SRC} -> {DST}")

    async with Store(DST) as store:
        table = store.chunks_table
        rows = await table.count_rows()
        # The declaration haiku.rag shipped before the change: default tokenizer.
        await table.create_index(
            "content_fts",
            config=FTS(with_position=True, remove_stop_words=False),
            replace=True,
        )
        for index in await table.list_indices():
            if index.index_type == "FTS":
                stats = await table.index_stats(index.name)
                print(f"  FTS {index.name}: covers {stats.num_indexed_rows}/{rows} rows")
        # Confirm the tokenizer really is the default one by asking a term the
        # ICU build finds and this one should not.
        from haiku.rag.store.repositories.chunk import ChunkRepository

        hits = await ChunkRepository(store).search(
            query="变分模态分解", limit=10, search_type="fts"
        )
        print(f"  probe '变分模态分解' -> {len(hits)} hits "
              f"(the ICU corpus returns many)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
