"""Near-misses by combination: both techniques here, never together.

Each case names two methods the corpus does use, and asks which paper uses both.
No chunk contains both, so the corpus cannot answer it — while nothing about the
question's surface says so, because neither term is foreign.

The pair is chosen and checked mechanically, from the chunk text:
  - each term appears in at least MIN_EACH chunks (so both look plausible)
  - no chunk contains both (so the answer does not exist)

That second condition is the whole point: a reader, and a model, cannot spot the
gap by recognising an unknown word. It has to be verified.
"""

import json
import re
import sys
from pathlib import Path

CJK = re.compile(r"[一-鿿]+")
# The question asks which PAPER uses both, so the corpus has to be checked at
# document level. A chunk-level check passes pairs that the same paper uses in
# different sections, which makes the question answerable after all.
MIN_EACH = 2
WANT = 20

# Techniques this literature uses. Both members of a pair must be present in the
# corpus for the question to look answerable.
CANDIDATES = [
    "变分模态分解", "经验模态分解", "集合经验模态", "小波变换", "注意力机制", "卷积神经网络",
    "长短期记忆", "门控循环单元", "随机森林", "支持向量机", "极限学习机", "遗传算法", "粒子群",
    "灰狼优化", "麻雀搜索", "鲸鱼优化", "迁移学习", "联邦学习", "生成对抗网络",
    "图神经网络", "Transformer", "自注意力", "残差网络", "数据增强", "贝叶斯优化", "主成分分析",
    "聚类", "相似日", "误差修正", "集成学习", "量子", "白鲸优化", "蚁群", "模拟退火", "小样本",
    "元学习", "时间卷积", "膨胀卷积", "双向长短期记忆", "对抗训练", "多任务学习", "知识蒸馏",
    "Autoformer", "Informer", "天气分型", "季节划分", "插值", "标准化", "归一化", "特征选择",
]

TEMPLATES = [
    "在收录的论文中，有没有哪一篇同时使用了{m}和{n}？如果有，是哪一篇？",
    "这批论文里，有论文把{m}和{n}结合起来用吗？请指出是哪篇以及怎么结合的。",
    "我想知道收录的文献中哪篇同时采用了{m}与{n}，它的具体做法是什么？",
]


def main(corpus_chunks: str, out: str) -> int:
    from haiku.rag.client import HaikuRAG
    from haiku.rag.config import AppConfig, set_config

    cfg = AppConfig()
    cfg.storage.auto_vacuum = False
    cfg.embeddings.model.name = "bge-m3"
    cfg.embeddings.model.vector_dim = 1024
    set_config(cfg)

    import asyncio

    async def load() -> list[str]:
        async with HaikuRAG(Path(corpus_chunks).resolve(), config=cfg) as rag:
            rows = await rag.store.chunks_table.query().select(["content", "document_id"]).to_list()
        return rows

    rows = asyncio.run(load())
    print(f"chunks: {len(rows)}")

    docs: dict[str, str] = {}
    for r in rows:
        docs[r["document_id"]] = docs.get(r["document_id"], "") + r["content"]
    holders: dict[str, set[str]] = {
        t: {d for d, text in docs.items() if t in text} for t in CANDIDATES
    }
    present = [t for t in CANDIDATES if len(holders[t]) >= MIN_EACH]
    print(f"terms used by >= {MIN_EACH} documents: {len(present)}")

    pairs = []
    for i, a in enumerate(present):
        for b in present[i + 1 :]:
            if holders[a] & holders[b]:
                continue  # a paper uses both, so it is answerable
            pairs.append((a, b, len(holders[a]) + len(holders[b])))
    pairs.sort(key=lambda p: -p[2])
    print(f"\npairs that never co-occur: {len(pairs)}")
    for a, b, _ in pairs[:WANT]:
        print(f"  {a} + {b}")

    cases = []
    for idx, (a, b, _) in enumerate(pairs[:WANT], 1):
        template = TEMPLATES[(idx - 1) % len(TEMPLATES)]
        cases.append({
            "id": f"c{idx:02d}",
            "answerable": False,
            "near_miss": True,
            "near_miss_kind": "combination",
            "question": template.format(m=a, n=b),
            "terms": [a, b],
            "verified_answerable": False,
            "verify_reason": (
                f"{a} in {len(holders[a])} documents, {b} in {len(holders[b])}, "
                "no document has both"
            ),
            "gold_answer": None,
            "gold_chunk_ids": [],
            "gold_document": None,
        })

    Path(out).write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cases), encoding="utf-8"
    )
    print(f"\nwrote {len(cases)} combination near-misses -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(
        sys.argv[1] if len(sys.argv) > 1 else ".data/corpus.lancedb",
        sys.argv[2] if len(sys.argv) > 2 else "lit_nearmiss_combo.jsonl",
    ))
