"""Compare two scored arms with the base's own paired statistics.

Reads two scored JSONL files, joins them on case id, and reports per-arm rates
plus exact McNemar on the paired verdicts. `mcnemar_exact`, `sign_test` and
`resolution` are imported from the base rather than reimplemented, so the
significance arithmetic is the one `evaluations pair` uses.
"""

import json
import statistics
import sys
from pathlib import Path

from evaluations.pairing import mcnemar_exact, resolution, sign_test


def load(path: str) -> dict[str, dict]:
    return {
        r["id"]: r
        for r in (
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def rate(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.1%})" if d else "—"


def subset(rows: dict[str, dict], pred) -> dict[str, dict]:
    return {k: v for k, v in rows.items() if pred(v)}


def report(name: str, rows: dict[str, dict]) -> None:
    ans = subset(rows, lambda r: r["answerable"])
    multi = subset(rows, lambda r: r.get("multihop"))
    unans = subset(rows, lambda r: not r["answerable"])
    correct = [r for r in ans.values() if r["answer_correct"]]
    cited = [r for r in ans.values() if r["cited_gold_chunk"]]
    refused_unans = [r for r in unans.values() if r["refused"]]
    print(f"\n  [{name}]  n={len(rows)}")
    print(f"    可答题正确率      {rate(len(correct), len(ans))}")
    print(f"    引用命中金标准    {rate(len(cited), len(ans))}")
    if multi:
        m_ok = sum(1 for r in multi.values() if r["answer_correct"])
        print(f"    多跳题正确率      {rate(m_ok, len(multi))}")
    print(f"    挖坑题拒答率      {rate(len(refused_unans), len(unans))}")
    print(f"    可答题误拒答      {sum(1 for r in ans.values() if r['refused'])}")
    searches = [r.get("n_search_queries") or 0 for r in rows.values()]
    print(f"    平均查询数        {statistics.mean(searches):.2f}")
    print(f"    平均耗时          {statistics.mean([r.get('elapsed_s') or 0 for r in rows.values()]):.1f}s")
    print(f"    运行报错          {sum(1 for r in rows.values() if r.get('error'))}")


def main(a_path: str, b_path: str) -> int:
    a, b = load(a_path), load(b_path)
    shared = sorted(set(a) & set(b))
    print(f"paired cases: {len(shared)}  (A={len(a)} B={len(b)})")
    report("A " + Path(a_path).stem, {k: a[k] for k in shared})
    report("B " + Path(b_path).stem, {k: b[k] for k in shared})

    # McNemar on the answer verdict, answerable cases only
    bb = cc = 0
    for k in shared:
        if not a[k]["answerable"]:
            continue
        av, bv = bool(a[k]["answer_correct"]), bool(b[k]["answer_correct"])
        if bv and not av:
            bb += 1
        elif av and not bv:
            cc += 1
    p = mcnemar_exact(bb, cc)
    disc = bb + cc
    need = resolution(disc)
    print(f"\n  === McNemar on correctness (answerable) ===")
    print(f"    B correct / A wrong : b = {bb}")
    print(f"    A correct / B wrong : c = {cc}")
    print(f"    discordant = {disc}   exact two-sided p = {p:.4f}"
          f"   {'SIGNIFICANT' if p < 0.05 else 'not significant'}")
    print(f"    最小显著 |b-c| 在这个不一致数下 = {need}")

    # Sign test on how much each arm searched
    up = down = 0
    for k in shared:
        sa, sb = a[k].get("n_search_queries") or 0, b[k].get("n_search_queries") or 0
        if sb > sa:
            up += 1
        elif sa > sb:
            down += 1
    if up + down:
        print(f"\n  === 查询次数 B vs A ===")
        print(f"    B 搜得更多 {up} 次, A 更多 {down} 次, "
              f"符号检验 p = {sign_test(up, down):.4f}")

    # Refusal on unanswerable
    ua = [k for k in shared if not a[k]["answerable"] and not a[k].get("near_miss")]
    if ua:
        ra = sum(1 for k in ua if a[k]["refused"])
        rb = sum(1 for k in ua if b[k]["refused"])
        print(f"\n  === 挖坑题拒答率 ===")
        print(f"    A {rate(ra, len(ua))}   B {rate(rb, len(ua))}")

    # The near-miss metric: did the model correctly say no such paper exists.
    nm = [k for k in shared if a[k].get("near_miss")]
    if nm:
        ha = sum(1 for k in nm if a[k].get("handled_correctly"))
        hb = sum(1 for k in nm if b[k].get("handled_correctly"))
        bb = cc = 0
        for k in nm:
            av, bv = bool(a[k].get("handled_correctly")), bool(b[k].get("handled_correctly"))
            if bv and not av:
                bb += 1
            elif av and not bv:
                cc += 1
        p = mcnemar_exact(bb, cc)
        print(f"\n  === 近失题:正确指出「没有论文同时使用两者」 ===")
        print(f"    A {rate(ha, len(nm))}   B {rate(hb, len(nm))}")
        print(f"    McNemar b={bb} c={cc} discordant={bb + cc} p={p:.4f}"
              f"   {'SIGNIFICANT' if p < 0.05 else 'not significant'}")
        print(f"    最小显著 |b-c| = {resolution(bb + cc)}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: compare_arms.py <baseline.scored.jsonl> <treatment.scored.jsonl>")
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
