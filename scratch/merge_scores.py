"""Fold a retry run's scored rows back over the main run's, dropping failures."""

import json
import sys
from pathlib import Path


def load(path: str) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def main(main_path: str, retry_path: str) -> int:
    main = {r["id"]: r for r in load(main_path)}
    replaced = 0
    for row in load(retry_path):
        if not row.get("error"):
            main[row["id"]] = row
            replaced += 1
    rows = sorted(main.values(), key=lambda r: r["id"])
    Path(main_path).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    errors = sum(1 for r in rows if r.get("error"))
    print(f"merged {len(rows)} cases, {replaced} replaced, {errors} still failing")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
