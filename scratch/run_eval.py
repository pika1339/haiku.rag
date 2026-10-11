"""Run one arm of the evaluation over the eval set, recording what happened.

Arms differ only in which capabilities are registered:
  plain            the RAG capability alone, which is what the base ships
  policy           + CitationPolicyCapability (forced grounding declaration)
  coverage         + CoverageCapability (forced per-part coverage declaration)
  policy_coverage  + both

Everything else — model, corpus, questions, request limit — is identical, so a
difference between arms is the capability.
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from pydantic_ai import Agent  # noqa: E402

from haiku.rag.capabilities.rag import create_capability as create_rag  # noqa: E402
from haiku.rag.config import AppConfig, set_config  # noqa: E402
from haiku.rag.utils.models import get_model  # noqa: E402
from lit_agent.coverage import create_capability as create_coverage  # noqa: E402

DB = Path(".data/corpus.lancedb").resolve()
REQUEST_LIMIT = 20


def config() -> AppConfig:
    cfg = AppConfig()
    cfg.storage.auto_vacuum = False
    cfg.embeddings.model.provider = "ollama"
    cfg.embeddings.model.name = "bge-m3"
    cfg.embeddings.model.vector_dim = 1024
    cfg.embeddings.batch_size = 128
    cfg.qa.model.provider = "openai"
    cfg.qa.model.name = "deepseek-flash"
    cfg.qa.model.base_url = "https://api.deepseek.com/v1"
    cfg.qa.model.api_key = os.environ.get("DEEPSEEK_API_KEY")
    cfg.qa.model.vision = False
    # `thinking: false` maps to reasoning_effort="none", which DeepSeek accepts
    # and which cuts a request from ~1.5s to ~0.6s.
    cfg.qa.model.thinking = False
    cfg.qa.max_searches = 4
    cfg.qa.max_executions = 6
    return cfg


@dataclass
class Deps:
    state: dict[str, Any] = field(default_factory=dict)


async def build_agent(arm: str, cfg: AppConfig):
    capabilities: list[Any] = [
        create_rag(db_path=DB, config=cfg, request_limit=REQUEST_LIMIT)
    ]
    if "policy" in arm:
        from haiku.rag.capabilities.policy import create_capability as create_policy

        capabilities.append(create_policy())
    if "coverage" in arm:
        capabilities.append(create_coverage())
    if "rewrite" in arm:
        from lit_agent.rewrite import create_capability as create_rewrite

        capabilities.append(create_rewrite())
    return Agent(get_model(cfg.qa.model, cfg), deps_type=Deps, capabilities=capabilities)


def snapshot(deps: Deps) -> dict[str, Any]:
    rag = deps.state.get("rag") or {}
    coverage = deps.state.get("coverage") or {}
    citations = rag.get("citation_index") or {}
    return {
        "cited_chunk_ids": list(citations),
        "searches": list((rag.get("searches") or {}).keys()),
        "n_search_queries": len(rag.get("searches") or {}),
        "coverage_declarations": coverage.get("declarations") or [],
        "citation_status": _citation_status(rag),
    }


def _citation_status(rag: dict) -> str | None:
    from haiku.rag.capabilities.ledger import CapabilityEvidenceRecord, citation_status

    record = CapabilityEvidenceRecord.model_validate(rag.get("evidence") or {})
    if record.question is None:
        return None
    return citation_status([record], question=record.question)


async def main(evalset_path: str, arm: str, out_path: str) -> int:
    load_dotenv(Path(".env"))
    cfg = config()
    set_config(cfg)

    cases = [json.loads(line) for line in Path(evalset_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    agent = await build_agent(arm, cfg)
    print(f"arm={arm}  cases={len(cases)}  model={cfg.qa.model.name} "
          f"thinking={cfg.qa.model.thinking}  request_limit={REQUEST_LIMIT}", flush=True)

    out = Path(out_path)
    t_start = time.time()
    with out.open("w", encoding="utf-8") as fh:
        for i, case in enumerate(cases, 1):
            deps = Deps()
            t0 = time.time()
            entry = {"id": case["id"], "arm": arm, "question": case["question"]}
            try:
                result = await agent.run(case["question"], deps=deps)
                entry["answer"] = str(result.output)
                entry["error"] = None
            except Exception as exc:  # noqa: BLE001
                entry["answer"] = ""
                entry["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            entry["elapsed_s"] = round(time.time() - t0, 1)
            entry.update(snapshot(deps))
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()
            flag = "ERR " if entry["error"] else "    "
            print(f"  [{i}/{len(cases)}] {flag}{entry['id']} {entry['elapsed_s']:5.1f}s "
                  f"searches={entry['n_search_queries']} cites={len(entry['cited_chunk_ids'])} "
                  f"cov={len(entry['coverage_declarations'])}", flush=True)
    print(f"\ndone in {time.time() - t_start:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: run_eval.py <evalset.jsonl> <arm> <out.jsonl>")
        sys.exit(2)
    sys.exit(asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3])))
