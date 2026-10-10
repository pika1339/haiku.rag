"""Pin down where the default 'simple' tokenizer drops a Chinese run.

Each test row is `MARK##，<Chinese run of length L>`. The ASCII marker survives
tokenization either way, so it proves the row is indexed; the exact Chinese run
query then shows whether that run survived as a token or was dropped.
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
FILL = "检"
LENGTHS = [4, 8, 12, 13, 14, 20, 30, 40, 41, 60]


def _boom(*args, **kwargs):
    raise AssertionError("embedder was called")


async def main() -> int:
    EmbedderWrapper.embed_query = _boom
    EmbedderWrapper.embed_documents = _boom

    cfg = AppConfig()
    cfg.storage.auto_vacuum = False
    set_config(cfg)

    root = Path(tempfile.mkdtemp(prefix="lit-agent-len-"))
    db = root / "len.lancedb"
    runs = {}
    texts = []
    for i, length in enumerate(LENGTHS):
        marker = f"MARK{i:02d}"
        run = ("前缀" + FILL * (length - 2))[:length]
        runs[length] = (marker, run)
        texts.append(f"{marker}，{run}")

    try:
        doc = DoclingDocument(name="len")
        for t in texts:
            doc.add_text(label=DocItemLabel.TEXT, text=t)

        async with HaikuRAG(db, config=cfg, create=True) as rag:
            await rag.import_document(
                doc,
                [
                    Chunk(content=t, embedding=[0.0] * VECTOR_DIM, order=i)
                    for i, t in enumerate(texts)
                ],
                uri="zh://len",
                title="长度探测",
            )
            repo = rag.chunk_repository
            table = rag.store.chunks_table

            rows = await (
                table.query().select(["content_fts"]).limit(len(texts)).to_list()
            )
            print("content_fts rows as stored:")
            for r in rows:
                print(f"  {r['content_fts']!r}")

            print("\nmarker query -> row indexed?  |  exact-run query -> run survived?")
            for length, (marker, run) in runs.items():
                marker_hits = await repo.search(query=marker, limit=5, search_type="fts")
                run_hits = await repo.search(query=run, limit=5, search_type="fts")
                nbytes = len(run.encode("utf-8"))
                print(
                    f"  L={length:>2} ({nbytes:>2} bytes UTF-8)  "
                    f"marker={'HIT ' if marker_hits else 'MISS'}  "
                    f"exact-run={'HIT ' if run_hits else 'MISS'}"
                )

            print("\nsame rows, exact-run query under base_tokenizer='icu':")
            from lancedb.index import FTS

            await table.create_index(
                "content_fts",
                config=FTS(with_position=True, remove_stop_words=False, base_tokenizer="icu"),
                replace=True,
            )
            repo._fts_coverage_checked = False
            for length, (marker, run) in runs.items():
                run_hits = await repo.search(query=run, limit=5, search_type="fts")
                print(f"  L={length:>2}  exact-run={'HIT ' if run_hits else 'MISS'}")
            return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
