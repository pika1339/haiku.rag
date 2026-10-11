"""Build the reusable corpus database from papers/.

Idempotent and resumable: a PDF already in the database under its URI is skipped,
so an interrupted run is restarted by running this again.

Duplicate files are collapsed by content hash, because the directory holds the
same paper twice under a " copy" name.
"""

import asyncio
import hashlib
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

from haiku.rag.client import HaikuRAG  # noqa: E402
from haiku.rag.config import AppConfig, set_config  # noqa: E402

DB = Path(".data/corpus.lancedb")


def config() -> AppConfig:
    cfg = AppConfig()
    cfg.storage.auto_vacuum = False
    cfg.embeddings.model.provider = "ollama"
    cfg.embeddings.model.name = "bge-m3"
    cfg.embeddings.model.vector_dim = 1024
    # Ollama refuses a bge-m3 request past ~256 inputs; the default 512 never gets
    # through. 128 leaves room for a cold model load too.
    cfg.embeddings.batch_size = 128
    # A document that takes longer than this is skipped rather than allowed to
    # hold up the whole corpus.
    cfg.processing.conversion_timeout = 180.0
    # A paper that references a remote image makes docling fetch it, and on a
    # machine behind a proxy that fetch hangs until the socket times out. These
    # PDFs carry their images inline, so the fetch buys nothing.
    cfg.processing.conversion_options.fetch_remote_images = False
    # OCR on CPU costs ~90s per paper and returns an empty result for a PDF that
    # already carries a text layer, which every paper here does. It also pushes
    # the slow ones past conversion_timeout.
    cfg.processing.conversion_options.do_ocr = False
    return cfg


def unique_pdfs(pdf_dir: Path) -> list[Path]:
    seen: dict[str, Path] = {}
    for path in sorted(pdf_dir.glob("*.pdf")):
        if " copy" in path.name:
            continue
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in seen:
            print(f"  skip duplicate of {seen[digest].name}: {path.name}")
            continue
        seen[digest] = path
    return sorted(seen.values())


async def main(pdf_dir: str) -> int:
    cfg = config()
    set_config(cfg)
    # LanceDB needs an absolute path.
    db = DB.resolve()

    pdfs = unique_pdfs(Path(pdf_dir))
    print(f"corpus: {len(pdfs)} unique PDF(s) -> {db}")

    t_start = time.time()
    async with HaikuRAG(db, config=cfg, create=True) as rag:
        existing = {
            d.uri for d in await rag.list_documents(limit=10_000) if d.uri
        }
        print(f"already in database: {len(existing)}")

        for i, pdf in enumerate(pdfs, 1):
            uri = str(pdf)
            if uri in existing:
                print(f"  [{i}/{len(pdfs)}] skip (already ingested) {pdf.stem[:48]}")
                continue
            # Split the stages so a slow corpus can be attributed: conversion is
            # docling, chunking is the tokenizer, import is embedding plus write.
            stamp = time.strftime("%H:%M:%S")
            try:
                t0 = time.time()
                docling_doc = await rag.convert(pdf)
                t_conv = time.time() - t0

                t0 = time.time()
                chunks = await rag.chunk(docling_doc)
                t_chunk = time.time() - t0

                t0 = time.time()
                doc = await rag.import_document(
                    docling_doc, chunks, uri=str(pdf), title=pdf.stem
                )
                t_import = time.time() - t0
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i}/{len(pdfs)}] {stamp} FAILED {pdf.stem[:40]}: "
                      f"{type(exc).__name__}: {str(exc)[:80]}", flush=True)
                continue
            print(f"  [{i}/{len(pdfs)}] {stamp} conv={t_conv:5.1f}s "
                  f"chunk={t_chunk:4.1f}s embed+write={t_import:6.1f}s "
                  f"n={len(chunks):>3}  {pdf.stem[:34]}", flush=True)

        docs = await rag.list_documents(limit=10_000)
        rows = await rag.store.chunks_table.count_rows()

        # Rows written after the FTS index was first built stay outside it until a
        # vacuum commits them: ingesting 30 documents leaves the index covering a
        # fraction of the table, and full-text search then returns only that
        # fraction, with nothing in the response saying so.
        for index in await rag.store.chunks_table.list_indices():
            if index.index_type == "FTS":
                stats = await rag.store.chunks_table.index_stats(index.name)
                print(f"before vacuum, FTS covers {stats.num_indexed_rows}/{rows}")
        await rag.vacuum()
        print(f"\ndocuments: {len(docs)}  chunks: {rows}")
        print(f"total elapsed: {time.time() - t_start:.0f}s")

        indices = await rag.store.chunks_table.list_indices()
        for index in indices:
            if index.index_type == "FTS":
                stats = await rag.store.chunks_table.index_stats(index.name)
                print(f"FTS index {index.name}: covers "
                      f"{stats.num_indexed_rows}/{rows} rows")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "papers")))
