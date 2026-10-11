"""Build near-miss questions: answerable-looking, but not answerable here.

A near-miss is made by taking a real question that IS answerable from the corpus
and swapping one concrete condition for one the corpus does not cover — another
weather type, another metric, another horizon. The question keeps the
terminology and the shape of the ones that are answerable, so nothing about its
surface says "look elsewhere".

Each case carries the term it swapped in, and the case is only kept when that
term is verifiably absent from the corpus. That check is the thing that makes
"the corpus cannot answer this" a fact rather than a guess.

The early unanswerable questions asked about topics plainly outside the field
(WRF parameterisation, Himawari-8), and every arm refused them: a ceiling. These
are meant to look answerable.
"""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
from dotenv import load_dotenv  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

MODEL = "deepseek-flash"
CJK = re.compile(r"[一-鿿]+")

SWAP_PROMPT = """下面是一个研究者会对一批中文光伏发电功率预测论文提的问题,以及论文里的真实答案。

问题:{question}
真实答案:{answer}

请把它改造成一个**近失问题**:问题读起来同样专业、同样合理,和上面那个像同一个研究者
问的,但它问的那个**具体条件/数据集/对比对象/数值**,在这批论文里**并不存在**。

要求:
- 保留原问题的句式、术语和具体程度,只换掉那个关键条件
- 换上的条件在光伏预测领域里必须是**真实存在且合理的**(别的天气类型、别的评价指标、
  别的时间尺度、别的对比方法、别的地区),只是这批论文没做——不要换成明显不相关的东西
- 不要留下任何"这个问题可能没有答案"的暗示,要像一个正常提问
- 只输出 JSON:{{"question": "...", "swapped_in": "你换进来的那个具体条件,只写这个词或短语"}}

换进来的词要能在语料里被逐字检查,所以请写成一个具体的词/短语(如"沙尘天气""决定系数冬
季预测""15分钟分辨率"),不要写整句话。"""


def parse(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


async def chat(client, prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            r = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=900,
                reasoning_effort="none",
            )
            return r.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"    retry {type(exc).__name__}", flush=True)
            await asyncio.sleep(2 * (attempt + 1))
    return ""


async def main(evalset: str, out: str) -> int:
    load_dotenv(Path(".env"))
    client = AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1"
    )
    cases = [
        json.loads(x)
        for x in Path(evalset).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    seeds = [c for c in cases if c["answerable"] and c.get("verified_answerable")]
    print(f"seeds: {len(seeds)} verified answerable questions", flush=True)

    # The absence check runs against everything that was ingested, not the eval
    # set's excerpts.
    from haiku.rag.client import HaikuRAG
    from haiku.rag.config import AppConfig, set_config

    cfg = AppConfig()
    cfg.storage.auto_vacuum = False
    cfg.embeddings.model.name = "bge-m3"
    cfg.embeddings.model.vector_dim = 1024
    set_config(cfg)
    async with HaikuRAG(Path(".data/corpus.lancedb").resolve(), config=cfg) as rag:
        rows = await rag.store.chunks_table.query().select(["content"]).to_list()
    corpus = "".join(r["content"] for r in rows)
    print(f"corpus text: {len(corpus)} chars", flush=True)

    kept: list[dict] = []
    rejected = 0
    t0 = time.time()
    for i, seed in enumerate(seeds, 1):
        if len(kept) >= 20:
            break
        raw = await chat(client, SWAP_PROMPT.format(
            question=seed["question"], answer=str(seed["gold_answer"])[:600]))
        got = parse(raw)
        if not isinstance(got, dict) or not got.get("question") or not got.get("swapped_in"):
            rejected += 1
            continue
        term = str(got["swapped_in"]).strip()
        # The check: the swapped-in condition must not appear in the corpus.
        if term and term in corpus:
            print(f"  reject (in corpus: {term!r}): {got['question'][:40]}", flush=True)
            rejected += 1
            continue
        kept.append({
            "id": f"n{len(kept) + 1:02d}",
            "answerable": False,
            "near_miss": True,
            "question": str(got["question"]).strip(),
            "swapped_in": term,
            "seed_id": seed["id"],
            "seed_question": seed["question"],
            "gold_answer": None,
            "gold_chunk_ids": [],
            "gold_document": None,
            "verified_answerable": False,
            "verify_reason": f"swapped in {term!r}, absent from the corpus",
        })
        print(f"  [{len(kept)}] {term:<18} {str(got['question'])[:52]}", flush=True)

    Path(out).write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in kept), encoding="utf-8"
    )
    print(f"\nkept {len(kept)}, rejected {rejected}, {time.time() - t0:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(
        sys.argv[1] if len(sys.argv) > 1 else "lit_evalset.jsonl",
        sys.argv[2] if len(sys.argv) > 2 else "lit_nearmiss.jsonl",
    )))
