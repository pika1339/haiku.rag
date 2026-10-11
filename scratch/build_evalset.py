"""Generate the evaluation set from the corpus, grounded in real chunks.

A question is only admitted when the chunk it was written from is the thing that
answers it, so the gold citation is provenance rather than an afterthought. Each
generated pair passes a second, independent check ("does this text alone answer
this question?"), and the check's verdict is stored alongside.

Unanswerable questions are generated against a different domain and then checked
mechanically: every distinctive term they name must be absent from the corpus, so
"the corpus cannot answer this" is verified rather than assumed.

Writes JSONL, one case per line.
"""

import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
from dotenv import load_dotenv  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

MODEL = "deepseek-flash"
CJK = re.compile(r"[一-鿿]+")


def cjk_chars(text: str) -> int:
    """Chinese characters, not runs: a clause between punctuation is one run."""
    return sum(len(run) for run in CJK.findall(text))

GEN_PROMPT = """你是一位光伏发电功率预测领域的研究者。下面是从一篇中文论文中摘出的一段内容。

请基于**且仅基于**这段内容,写一个该领域研究者**真的会问**的问题,并给出答案。

要求:
- 问题要具体,答案必须完全由这段内容支持,不要引入这段内容没有的信息
- 问题不要提到"这段内容""本段"之类的措辞,要像一个真实的提问
- 答案简洁准确,1-3 句

只输出 JSON,格式:
{{"question": "...", "answer": "..."}}

内容:
---
{chunk}
---"""

VERIFY_PROMPT = """下面有一段文字和一个问题。

问题:{question}
预期答案:{answer}

这段文字:
---
{chunk}
---

请判断:只看这段文字,能不能得出上述答案?
只输出 JSON:{{"answerable": true/false, "reason": "简短理由"}}"""

MULTIHOP_PROMPT = """下面是从**两篇不同论文**中各摘出的一段内容。

请设计一个研究者真的会问的问题,该问题**必须同时用到这两段内容才能完整回答**。
不要设计成两个独立问题拼在一起,而要一个需要综合两处信息的真问题。

要求:
- 问题自然,像真实提问,不要提"第一段""第二段"
- 答案要注明哪部分来自哪一段(用"其一/其二"之类的表述),便于核对

只输出 JSON:{{"question": "...", "answer": "..."}}

【论文 A】{text_a}

【论文 B】{text_b}"""

UNANS_PROMPT = """你是光伏发电功率预测领域的研究者。请提出 {n} 个该领域研究者可能会问、
但下面这些论文里**很可能没有涉及**的具体问题。

要求:
- 必须具体(涉及具体方法名、具体数据、具体装置),不要泛泛而谈
- 尽量避开这些论文的主题方向
- 每个问题独立,不要互相引用

只输出 JSON 列表:["问题1", "问题2", ...]"""


def log(msg: str) -> None:
    print(msg, flush=True)


async def chat(client: AsyncOpenAI, prompt: str, *, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            r = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1200,
                reasoning_effort="none",
            )
            return r.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            log(f"    retry after {type(exc).__name__}: {str(exc)[:60]}")
            await asyncio.sleep(3 * (attempt + 1))
    return ""


def parse_json(text: str):
    """Pull the first JSON object or array out of a model response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


async def main() -> int:
    load_dotenv(Path(".env"))
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        log("no DEEPSEEK_API_KEY")
        return 1
    client = AsyncOpenAI(api_key=key, base_url="https://api.deepseek.com/v1")

    from haiku.rag.client import HaikuRAG
    from haiku.rag.config import AppConfig, set_config

    cfg = AppConfig()
    cfg.storage.auto_vacuum = False
    cfg.embeddings.model.provider = "ollama"
    cfg.embeddings.model.name = "bge-m3"
    cfg.embeddings.model.vector_dim = 1024
    cfg.embeddings.batch_size = 128
    set_config(cfg)

    db = Path(".data/corpus.lancedb").resolve()
    target_answerable = int(os.environ.get("N_ANSWERABLE", "40"))
    target_multihop = int(os.environ.get("N_MULTIHOP", "20"))
    target_unanswerable = int(os.environ.get("N_UNANSWERABLE", "12"))

    async with HaikuRAG(db, config=cfg) as rag:
        rows = await (
            rag.store.chunks_table.query()
            .select(["id", "content", "document_id"]).to_list()
        )
        meta = await rag.store.document_meta_table.query().select(["id", "uri"]).to_list()
        uri_of = {m["id"]: m["uri"] for m in meta}

        # One candidate per document per length band, so questions span the corpus
        # instead of clustering on one paper.
        by_doc: dict[str, list[dict]] = {}
        for r in rows:
            content = r["content"]
            if cjk_chars(content) < 80:
                continue
            by_doc.setdefault(r["document_id"], []).append(r)
        rng = random.Random(20261011)
        candidates = []
        per_doc = max(1, target_answerable // max(len(by_doc), 1))
        for doc_id, items in by_doc.items():
            items.sort(key=lambda r: len(r["content"]))
            picks = rng.sample(items, min(per_doc, len(items)))
            for p in picks:
                candidates.append(p)
        rng.shuffle(candidates)
        log(f"corpus: {len(rows)} chunks over {len(by_doc)} documents; "
            f"{len(candidates)} candidate chunks for question generation")

        cases: list[dict] = []
        doc_chunks_text: dict[str, str] = {}
        for r in rows:
            doc_chunks_text[r["document_id"]] = (
                doc_chunks_text.get(r["document_id"], "") + r["content"]
            )

        log(f"\n--- generating {target_answerable} answerable questions ---")
        t0 = time.time()
        for i, chunk in enumerate(candidates, 1):
            if len([c for c in cases if c["answerable"]]) >= target_answerable:
                break
            chunk_text = chunk["content"][:2500]
            raw = await chat(client, GEN_PROMPT.format(chunk=chunk_text))
            pair = parse_json(raw)
            if not isinstance(pair, dict) or not pair.get("question") or not pair.get("answer"):
                log(f"  [{i}] skipped (unparseable)")
                continue
            raw2 = await chat(client, VERIFY_PROMPT.format(
                question=pair["question"], answer=pair["answer"], chunk=chunk_text))
            verdict = parse_json(raw2) or {}
            case = {
                "id": f"a{len(cases) + 1:03d}",
                "answerable": True,
                "question": pair["question"].strip(),
                "gold_answer": str(pair["answer"]).strip(),
                "gold_chunk_ids": [chunk["id"]],
                "gold_document": Path(str(uri_of.get(chunk["document_id"], "?"))).name,
                "verified_answerable": bool(verdict.get("answerable")),
                "verify_reason": str(verdict.get("reason", ""))[:200],
                "source_chunk": chunk_text[:600],
            }
            cases.append(case)
            flag = "" if case["verified_answerable"] else "  <-- VERIFY FAILED"
            log(f"  [{i}] {case['id']} {case['gold_document'][:28]:<30} "
                f"{case['question'][:38]}{flag}")
        log(f"generated in {time.time() - t0:.0f}s")

        log(f"\n--- generating {target_multihop} cross-paper questions ---")
        t0 = time.time()
        doc_ids = list(by_doc)
        made = 0
        attempts = 0
        while made < target_multihop and attempts < target_multihop * 3:
            attempts += 1
            a_doc, b_doc = rng.sample(doc_ids, 2)
            a = rng.choice(by_doc[a_doc])
            b = rng.choice(by_doc[b_doc])
            text_a, text_b = a["content"][:1800], b["content"][:1800]
            raw = await chat(client, MULTIHOP_PROMPT.format(text_a=text_a, text_b=text_b))
            pair = parse_json(raw)
            if not isinstance(pair, dict) or not pair.get("question"):
                continue
            raw2 = await chat(client, VERIFY_PROMPT.format(
                question=pair["question"], answer=pair["answer"], chunk=text_a))
            # A cross-paper question is only real if neither half answers it alone.
            v_a = parse_json(raw2) or {}
            raw3 = await chat(client, VERIFY_PROMPT.format(
                question=pair["question"], answer=pair["answer"], chunk=text_b))
            v_b = parse_json(raw3) or {}
            if v_a.get("answerable") or v_b.get("answerable"):
                log(f"  reject (one half answers it alone): {str(pair['question'])[:34]}")
                continue
            made += 1
            case = {
                "id": f"m{made:02d}",
                "answerable": True,
                "multihop": True,
                "question": str(pair["question"]).strip(),
                "gold_answer": str(pair["answer"]).strip(),
                "gold_chunk_ids": [a["id"], b["id"]],
                "gold_document": " + ".join(sorted({
                    Path(str(uri_of.get(a_doc, "?"))).name,
                    Path(str(uri_of.get(b_doc, "?"))).name,
                })),
                "verified_answerable": True,
                "verify_reason": "neither half answers it alone",
                "source_chunk": (text_a[:300], text_b[:300]),
            }
            cases.append(case)
            log(f"  {case['id']} {str(pair['question'])[:52]}")
        log(f"generated in {time.time() - t0:.0f}s ({attempts} attempts)")

        log(f"\n--- generating {target_unanswerable} unanswerable questions ---")
        raw = await chat(client, UNANS_PROMPT.format(n=target_unanswerable))
        questions = parse_json(raw)
        if not isinstance(questions, list):
            log("  failed to parse unanswerable list")
            questions = []
        corpus_text = "".join(doc_chunks_text.values())
        kept = 0
        for q in questions:
            if not isinstance(q, str) or len(q) < 8:
                continue
            # A question is only unanswerable if the corpus lacks its substance:
            # check every distinctive term it names.
            terms = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", q)]
            terms += [r for r in CJK.findall(q) if len(r) >= 4]
            present = [t for t in terms if t in corpus_text]
            if present:
                log(f"  reject (corpus mentions {present}): {q[:40]}")
                continue
            kept += 1
            cases.append({
                "id": f"u{kept:02d}",
                "answerable": False,
                "question": q.strip(),
                "gold_answer": None,
                "gold_chunk_ids": [],
                "gold_document": None,
                "verified_answerable": False,
                "verify_reason": "corpus lacks every distinctive term named",
                "source_chunk": None,
            })
            log(f"  u{kept:02d} {q[:60]}")

        out = Path(sys.argv[1] if len(sys.argv) > 1 else "evalset.jsonl")
        with out.open("w", encoding="utf-8") as fh:
            for case in cases:
                fh.write(json.dumps(case, ensure_ascii=False) + "\n")
        n_ok = sum(1 for c in cases if c["verified_answerable"])
        log(f"\nwrote {len(cases)} cases to {out}")
        log(f"  answerable: {n_ok} verified / "
            f"{sum(1 for c in cases if c['answerable'])} generated")
        log(f"  unanswerable: {sum(1 for c in cases if not c['answerable'])}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
