"""Exploratory: how often does a retrieval failure still end in a correct answer?

The hypothesis came out of five cases in §4.7, so this is not a pre-registered
test — it is the same data read at a larger n, and it is labelled exploratory
throughout.

"Retrievable" is a property of the corpus and the retriever, not of any run: it
asks whether the question itself brings the gold chunk back in the top 20. A case
whose gold chunk never comes back but whose answer is still correct is a
retrieval failure the model absorbed.

Reports both strict (every gold chunk found) and loose (at least one found),
because a cross-paper case has two gold chunks and the two readings differ.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

LIMIT = 20


async def main(evalset: str, results: str) -> int:
    from haiku.rag.client import HaikuRAG
    from haiku.rag.config import AppConfig, set_config

    cfg = AppConfig()
    cfg.embeddings.model.name = "bge-m3"
    cfg.embeddings.model.vector_dim = 1024
    cfg.embeddings.batch_size = 128
    set_config(cfg)

    cases = [
        json.loads(x)
        for x in Path(evalset).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    scored = {
        json.loads(x)["id"]: json.loads(x)
        for x in Path(results).read_text(encoding="utf-8").splitlines()
        if x.strip()
    }
    answerable = [c for c in cases if c["answerable"]]
    print(f"answerable cases: {len(answerable)}  (of {len(cases)} total)")

    rows = []
    async with HaikuRAG(Path(".data/corpus.lancedb").resolve(), config=cfg) as rag:
        for c in answerable:
            gold = set(c.get("gold_chunk_ids") or [])
            if not gold:
                continue
            hits = await rag.search(c["question"], limit=LIMIT, include_images=False)
            found_ids = {getattr(h, "chunk_id", None) for h in hits}
            n_found = len(gold & found_ids)
            rows.append({
                "id": c["id"],
                "multihop": bool(c.get("multihop")),
                "n_gold": len(gold),
                "n_found": n_found,
                "strict": n_found == len(gold),
                "loose": n_found > 0,
                "correct": bool((scored.get(c["id"]) or {}).get("answer_correct")),
            })

    def table(pred, label):
        miss = [r for r in rows if not pred(r)]
        hit = [r for r in rows if pred(r)]
        m_ok = sum(1 for r in miss if r["correct"])
        h_ok = sum(1 for r in hit if r["correct"])
        print(f"\n--- {label} ---")
        print(f"  retrievable     n={len(hit):>3}  correct={h_ok:>3} "
              f"({h_ok / len(hit):.1%})" if hit else "  retrievable     n=0")
        if miss:
            print(f"  NOT retrievable n={len(miss):>3}  correct={m_ok:>3} "
                  f"({m_ok / len(miss):.1%})   <-- 自愈率")
        return len(miss), m_ok

    n_miss, m_ok = table(lambda r: r["strict"], "strict: every gold chunk in top-20")
    table(lambda r: r["loose"], "loose: at least one gold chunk in top-20")

    print("\n--- 逐题 ---")
    for r in rows:
        flag = "" if r["strict"] else ("  <-- 检索失败但答对" if r["correct"] else "  <-- 检索失败且答错")
        print(f"  {r['id']} {'多跳' if r['multihop'] else '单块'} "
              f"gold={r['n_gold']} found={r['n_found']} correct={r['correct']}{flag}")

    print(f"\n【探索性结论】检索失败 {n_miss} 题,其中答对 {m_ok} 题 "
          f"= {m_ok / n_miss:.1%}" if n_miss else "无检索失败")
    print("注:假设来自 §4.7 的 5 个案例,本分析是同一批数据在更大 n 上的重读,")
    print("    属于探索性分析,不是预先注册的检验。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(
        sys.argv[1] if len(sys.argv) > 1 else "lit_evalset.jsonl",
        sys.argv[2] if len(sys.argv) > 2 else ".data/runs/scored_plain.jsonl",
    )))
