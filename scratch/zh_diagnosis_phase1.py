"""Chinese-corpus diagnosis, phase 1: is the PDF parsed correctly?

Runs the REAL pipeline (docling-local conversion, docling-local chunking) and
stores the result, with only the embedding replaced by a zero-vector stub so the
parse check does not wait on Ollama. Then it reads the segments back OUT OF THE
DATABASE and prints them, which is the question this phase answers: a retrieval
failure must not be blamed on tokenization when the stored text is already junk.

Phase 2 (real embeddings, vector/hybrid comparison) is a separate run.
"""

import asyncio
import re
import shutil
import statistics
import sys
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, set_config
from haiku.rag.embeddings import EmbedderWrapper

CJK = re.compile(r"[一-鿿]")
# The default 'simple' FTS tokenizer splits on whitespace and punctuation, so a
# contiguous CJK run is one token, and LanceDB drops tokens over 40 UTF-8 bytes.
RUN_LIMIT_BYTES = 40
SAMPLES = 3


def stub_embedder() -> None:
    """Every vector becomes zeros: FTS never reads one, and cosine is never used."""

    async def fake_documents(self, texts):
        return [[0.0] * self._vector_dim for _ in texts]

    async def fake_query(self, text):
        return [[0.0] * self._vector_dim]

    EmbedderWrapper._embed_documents = fake_documents
    EmbedderWrapper._embed_query = fake_query


def clip(text: str, n: int = 120) -> str:
    flat = text.replace("\n", " ⏎ ")
    return flat[:n] + ("…" if len(flat) > n else "")


def report_text_quality(label: str, contents: list[str]) -> None:
    total = sum(len(c) for c in contents)
    empty = sum(1 for c in contents if not c.strip())
    cjk = sum(len(CJK.findall(c)) for c in contents)
    fffd = sum(c.count("�") for c in contents)
    control = Counter(
        ch for c in contents for ch in c if unicodedata.category(ch) == "Cc" and ch not in "\n\r\t"
    )
    lengths = [len(c) for c in contents] or [0]
    print(f"[{label}] chunks={len(contents)} chars={total} empty={empty}")
    print(
        f"[{label}] CJK chars={cjk} ({cjk / total:.1%} of all chars) "
        f"U+FFFD(replacement)={fffd} control-chars={dict(control)}"
    )
    print(
        f"[{label}] chunk length chars: min={min(lengths)} "
        f"median={statistics.median(lengths):.0f} max={max(lengths)}"
    )


def report_cliff(label: str, contents: list[str]) -> None:
    """How much of the corpus the 40-byte token limit silently drops."""
    total_ck = dropped_ck = 0
    runs_over = 0
    runs_total = 0
    for text in contents:
        for run in re.findall(r"[一-鿿]+", text):
            n, nbytes = len(run), len(run.encode("utf-8"))
            runs_total += 1
            total_ck += n
            if nbytes > RUN_LIMIT_BYTES:
                runs_over += 1
                dropped_ck += n
    if not total_ck:
        print(f"[{label}] no CJK")
        return
    print(
        f"[{label}] CJK runs={runs_total}, over 40 bytes={runs_over} "
        f"({runs_over / runs_total:.1%})"
    )
    print(
        f"[{label}] CJK chars inside over-40-byte runs = {dropped_ck}/{total_ck} "
        f"= {dropped_ck / total_ck:.1%}  <-- invisible to FTS under 'simple'"
    )


async def main(pdf_dir: str) -> int:
    stub_embedder()

    cfg = AppConfig()
    cfg.storage.auto_vacuum = False
    set_config(cfg)

    pdfs = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdfs:
        print(f"no PDFs in {pdf_dir}")
        return 1

    root = Path(tempfile.mkdtemp(prefix="lit-agent-zh1-"))
    db = root / "zh.lancedb"
    print(f"db: {db}\nparsing {len(pdfs)} PDF(s) with processing.converter="
          f"{cfg.processing.converter}, chunker={cfg.processing.chunker}, "
          f"chunk_size={cfg.processing.chunk_size}, "
          f"tokenizer={cfg.processing.chunking_tokenizer}\n")

    try:
        async with HaikuRAG(db, config=cfg, create=True) as rag:
            for pdf in pdfs:
                print("=" * 78)
                print(f"### {pdf.name}")
                try:
                    docling_doc, chunks = await rag.create_document_from_source(pdf)
                except Exception as exc:  # noqa: BLE001
                    print(f"  CONVERSION FAILED: {type(exc).__name__}: {exc}")
                    continue
                print(f"  document_id={docling_doc.id} chunks={len(chunks)}")

            # ---- read the segments back OUT OF THE DATABASE --------------------
            rows = await (
                rag.store.chunks_table.query()
                .select(["document_id", "content", "content_fts"])
                .to_list()
            )
            print("\n" + "=" * 78)
            print("STORED SEGMENTS READ BACK FROM THE DATABASE")
            print("=" * 78)
            contents = [r["content"] for r in rows]
            fts_cols = [r["content_fts"] for r in rows]
            report_text_quality("stored", contents)
            report_text_quality("content_fts", fts_cols)
            report_cliff("stored", contents)

            print(f"\n--- {SAMPLES} stored segments, verbatim (repr) ---")
            for i in range(0, min(SAMPLES, len(contents))):
                print(f"\n[{i}] content     = {clip(contents[i], 220)!r}")
                print(f"[{i}] content_fts = {clip(fts_cols[i], 220)!r}")

            print(f"\n--- {SAMPLES} stored segments, verbatim (rendered) ---")
            for i in range(0, min(SAMPLES, len(contents))):
                print(f"\n[{i}] {clip(contents[i], 300)}")

            print("\n--- longest stored segments (most likely to be unsearchable) ---")
            order = sorted(range(len(contents)), key=lambda i: -len(contents[i]))
            for i in order[:SAMPLES]:
                runs = re.findall(r"[一-鿿]+", contents[i])
                longest = max(runs, key=len) if runs else ""
                print(
                    f"\n[{i}] {len(contents[i])} chars; longest CJK run "
                    f"{len(longest)} chars / {len(longest.encode())} bytes"
                    f"{'  <-- DROPPED' if len(longest.encode()) > RUN_LIMIT_BYTES else ''}"
                )
                print(f"     {clip(contents[i], 200)}")
            return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "papers")))
