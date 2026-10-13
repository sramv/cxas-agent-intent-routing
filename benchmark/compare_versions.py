#!/usr/bin/env python3
"""Compares Version 1 (LLM Router) vs Version 2 (DiffusionGemma on Cloud Run) benchmark results
and generates the final side-by-side markdown report."""

import json
import statistics
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
REPORT_PATH = BASE_DIR.parent / "reports" / "experiment_report.md"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    v1_sum_path = RESULTS_DIR / "v1_standard_summary.json"
    v2_sum_path = RESULTS_DIR / "v2_standard_summary.json"
    dg_sum_path = RESULTS_DIR / "diffusion_gemma_direct_summary.json"

    if not v1_sum_path.exists():
        raise FileNotFoundError(f"Missing {v1_sum_path}")

    v1 = load_json(v1_sum_path)
    v2 = load_json(v2_sum_path) if v2_sum_path.exists() else None
    dg = load_json(dg_sum_path) if dg_sum_path.exists() else None

    print("Loaded V1 summary:", v1["routing_accuracy_pct"], "%")
    if v2:
        print("Loaded V2 summary:", v2["routing_accuracy_pct"], "%")
    if dg:
        print(
            "Loaded Direct DiffusionGemma summary: sub-agent acc =",
            dg["sub_agent_accuracy_pct"],
            "%, 160-intent acc =",
            dg["intent_accuracy_pct"],
            "%",
        )


if __name__ == "__main__":
    main()
