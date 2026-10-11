# 研究脚本

这些脚本产出 `PLAN.md` 第三节的全部数字。它们只做本地检索与 LanceDB 读取，
不调用任何付费接口，可以直接重跑。

运行方式（从仓库根目录）：

```bash
uv run python scratch/<脚本>.py papers
```

前置：中文论文 PDF 放在 `papers/`（PDF 本体不进仓库，清单见 `papers/SOURCES.md`）。
真实向量实验另外需要 Ollama 上的 `bge-m3`。

| 脚本 | 回答什么问题 | 主要产出 |
|---|---|---|
| `fts_probe.py` | 中文查询在默认 FTS 索引下能不能命中 | 英文全中；中文只有整段一字不差地当查询才命中 |
| `fts_len_probe.py` | `max_token_length` 的边界在哪 | 连续中文 13 字（39 字节）命中；14 字（42 字节）落空 |
| `hybrid_probe.py` | 向量侧漏检时，hybrid 能不能靠 FTS 救回 | 英文从 rank 4 拉到 rank 1；中文与纯向量逐位相同 |
| `zh_diagnosis_phase1.py` | 真实 PDF 的解析质量（把解析问题和检索问题分开） | 220 段，0 空段、0 替换符、0 控制符 |
| `zh_diagnosis_phase2.py` | 真实语料上 FTS 的 ground-truth 命中率 | 默认索引 zero-hit 97.5%→0%，MAP 0.0000→0.5714（40 个自动抽取词） |
| `fts_ab_benchmark.py` | 改前/改后的 MAP / Recall / nDCG，加英文对照组 | 中文 MAP 0.2254→0.6989；英文两组全部无指标下降 |
| `pdf_meta.py` | 从 PDF 首页抽标题/作者/出处 | `papers/SOURCES.md` 的来源 |
| `build_corpus.py` | 建可复用语料库(幂等、可续跑、入库后 vacuum) | `.data/corpus.lancedb`,30 篇 / 2825 块 |
| `build_evalset.py` | 生成评测集:单块题 + 跨篇多跳题 + 挖坑题,带二次校验 | `lit_evalset.jsonl`,57 题 |
| `run_eval.py` | 跑某一臂:`plain` / `policy` / `coverage` / `rewrite` | 逐题答案、引用、查询数、覆盖度声明 |
| `score_eval.py` | 判官评分 + 机械引用核对 | 每题的 `answer_correct` / `refused` / `cited_gold_chunk` |
| `compare_arms.py` | 两臂配对统计(复用底座 `mcnemar_exact`) | 正确率、引用命中、拒答率、McNemar、符号检验 |
| `merge_scores.py` | 把补跑结果合并回主结果 | — |
| `baseline_vector.py` | 真实 bge-m3 向量下的三臂对照 | 见 `FINDINGS.md` §2 |

## 方法说明

**ground truth 是「该词逐字出现在某个 chunk 中」。** 因此这些指标量的是**可检索性**，
不是语义相关性——换 tokenizer 能改变的正是前者。

`fts_ab_benchmark.py` 里 MAP 直接复用底座
`evaluations/evaluators/citation.py::average_precision`；recall 与 nDCG 复刻
`evaluations/evaluators/retrieval.py` 的公式，但**粒度是 chunk 而非文档**
（三篇中文文档撑不起文档级检索指标）。

三个探针脚本在类级别把 `EmbedderWrapper` 的嵌入方法换成抛异常的桩，
所以「全程没有调用嵌入模型」是可验证的，而不是靠约定。
