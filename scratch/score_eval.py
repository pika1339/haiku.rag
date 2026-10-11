"""Score one arm's results against the eval set.

Two verdicts per case, on top of what the run recorded:

- `answer_correct`, for answerable cases: an LLM judge against the gold answer.
- `refused`, for every case: whether the answer declines to answer. Correct
  behaviour is the opposite of the case's answerability, so a refusal on an
  answerable case and an answer on an unanswerable one are both failures.

Citations are checked mechanically: whether a chunk id from the gold set, or a
chunk of the gold document, appears among the cited ids.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
from dotenv import load_dotenv  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

JUDGE_MODEL = "deepseek-flash"

JUDGE_ANSWER = """判断下面两个答案是否等价。

问题:{question}
标准答案:{gold}
待判答案:{got}

只要待判答案包含标准答案的实质信息即可判为等价,措辞不同、多说了无关内容都不影响。
如果待判答案说找不到/无法回答,或者与标准答案的实质信息不符,判为不等价。

只输出 JSON:{{"equivalent": true/false, "reason": "简短理由"}}"""

JUDGE_REFUSAL = """判断下面这段回答是否在"拒绝回答"——即明确表示信息不足、找不到、无法从这些文档中回答,
而不是给出了实质性答案。

回答:{answer}

只输出 JSON:{{"refused": true/false}}"""

# A near-miss case asks about a combination no paper uses. Plain refusal is not the
# only correct shape: saying "no paper does this" answers the question correctly,
# and a refusal judge does not see that as a refusal.
JUDGE_NEARMISS = """下面有一个关于一批论文的问题,以及模型给出的回答。

问题:{question}
回答:{answer}

正确答案是:这批论文里【没有】任何一篇同时具备问题中提到的两点。

请判断回答是否正确地表达了这一点——即明确说明**没有这样的论文**,或者表示无法回答。
如果回答声称存在这样一篇论文、或指名了具体论文,判为不正确。
如果回答绕开问题、没有明确表态,也判为不正确。

只输出 JSON:{{"correct": true/false, "reason": "简短理由"}}"""


def parse(text: str):
    import re

    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


async def ask(client, prompt: str) -> dict:
    for attempt in range(3):
        try:
            r = await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=400,
                reasoning_effort="none",
            )
            return parse(r.choices[0].message.content or "") or {}
        except Exception:  # noqa: BLE001
            if attempt == 2:
                return {}
            await asyncio.sleep(2 * (attempt + 1))
    return {}


async def main(results_path: str, evalset_path: str, out_path: str) -> int:
    load_dotenv(Path(".env"))
    client = AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1"
    )
    results = [json.loads(x) for x in Path(results_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    gold = {c["id"]: c for c in (json.loads(x) for x in Path(evalset_path).read_text(encoding="utf-8").splitlines() if x.strip())}

    scored = []
    for r in results:
        case = gold[r["id"]]
        answer = (r.get("answer") or "").strip()
        entry = dict(r)
        if case.get("near_miss"):
            # The metric for these is whether the model correctly says no such
            # paper exists, which is neither "correct answer" nor "refusal".
            if not answer:
                entry["handled_correctly"] = False
                entry["judge_reason"] = "no answer"
            else:
                v = await ask(client, JUDGE_NEARMISS.format(
                    question=case["question"], answer=answer))
                entry["handled_correctly"] = bool(v.get("correct"))
                entry["judge_reason"] = str(v.get("reason", ""))[:200]
            entry["refused"] = bool(entry.get("refused"))
            entry["near_miss"] = True
            entry["answer_correct"] = None
            entry.update(
                cited_gold_chunk=False, gold_document=None, answerable=False,
                gold_answer=None, verified_answerable=False,
            )
            scored.append(entry)
            print(f"  {entry['id']} handled={entry['handled_correctly']} "
                  f"{str(entry.get('judge_reason',''))[:44]}", flush=True)
            continue
        if not answer:
            entry.update(answer_correct=False, refused=False, judge_reason="no answer")
        else:
            refusal = await ask(client, JUDGE_REFUSAL.format(answer=answer))
            entry["refused"] = bool(refusal.get("refused"))
            if case["answerable"]:
                v = await ask(client, JUDGE_ANSWER.format(
                    question=case["question"], gold=case["gold_answer"], got=answer))
                entry["answer_correct"] = bool(v.get("equivalent"))
                entry["judge_reason"] = str(v.get("reason", ""))[:200]
            else:
                entry["answer_correct"] = None
                entry["judge_reason"] = "unanswerable case"
        # citation, mechanical
        gold_docs = {case["gold_document"]} if case.get("gold_document") else set()
        entry["cited_gold_chunk"] = bool(
            set(case.get("gold_chunk_ids") or []) & set(r.get("cited_chunk_ids") or [])
        )
        entry["gold_document"] = case.get("gold_document")
        entry["answerable"] = case["answerable"]
        entry["gold_answer"] = case.get("gold_answer")
        entry["verified_answerable"] = case.get("verified_answerable")
        scored.append(entry)
        print(f"  {entry['id']} correct={entry['answer_correct']} refused={entry['refused']} "
              f"citedGold={entry['cited_gold_chunk']}", flush=True)

    Path(out_path).write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in scored), encoding="utf-8"
    )

    ans = [s for s in scored if s["answerable"]]
    unans = [s for s in scored if not s["answerable"]]
    print(f"\n=== {Path(results_path).stem} ===")
    print(f"  answerable    n={len(ans):>3}  correct={sum(1 for s in ans if s['answer_correct']):>3} "
          f"({sum(1 for s in ans if s['answer_correct']) / max(len(ans), 1):.1%})"
          f"  refused={sum(1 for s in ans if s['refused'])}")
    print(f"  cited gold    n={len(ans):>3}  hit={sum(1 for s in ans if s['cited_gold_chunk']):>3} "
          f"({sum(1 for s in ans if s['cited_gold_chunk']) / max(len(ans), 1):.1%})")
    print(f"  unanswerable  n={len(unans):>3}  refused={sum(1 for s in unans if s['refused']):>3} "
          f"({sum(1 for s in unans if s['refused']) / max(len(unans), 1):.1%})")
    near = [s for s in scored if s.get("near_miss")]
    if near:
        ok = sum(1 for s in near if s.get("handled_correctly"))
        print(f"  near-miss     n={len(near):>3}  handled correctly={ok:>3} ({ok / len(near):.1%})")
    print(f"  errors        n={sum(1 for s in scored if s.get('error'))}")
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: score_eval.py <results.jsonl> <evalset.jsonl> <scored.jsonl>")
        sys.exit(2)
    sys.exit(asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3])))
