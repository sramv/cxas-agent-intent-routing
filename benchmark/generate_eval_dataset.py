#!/usr/bin/env python3
"""Generates a stratified evaluation dataset across all 160 Telco Intents for CXAS V1 vs V2 benchmarking."""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TAXONOMY_PATH = BASE_DIR.parent / "taxonomy" / "telco_160_intents.json"
OUT_PATH = BASE_DIR / "telco_eval_dataset.jsonl"


def generate_variations(intent_item: dict) -> list[dict]:
    """Creates 3 stratified test utterances per intent (480 total across 160 intents):
    1. direct: Clear, canonical phrasing from sample_utterance
    2. conversational: Realistic natural customer phrasing with context
    3. concise: Short, urgent mobile-app style query
    """
    intent_id = intent_item["intent_id"]
    intent_name = intent_item["intent_name"]
    sub_agent = intent_item["sub_agent"]
    sample = intent_item["sample_utterance"]
    desc = intent_item["description"]

    # Clean lowercase description clause for natural synthesis
    desc_clause = desc
    if desc_clause.lower().startswith("user "):
        desc_clause = desc_clause[5:]

    cases = [
        {
            "case_id": f"{intent_id}_direct",
            "intent_id": intent_id,
            "intent_name": intent_name,
            "expected_sub_agent": sub_agent,
            "difficulty": "standard",
            "utterance": sample,
        },
        {
            "case_id": f"{intent_id}_conversational",
            "intent_id": intent_id,
            "intent_name": intent_name,
            "expected_sub_agent": sub_agent,
            "difficulty": "conversational",
            "utterance": f"Hi there, I need help with {intent_name.lower()}. Specifically, { sample[0].lower() + sample[1:] }",
        },
        {
            "case_id": f"{intent_id}_concise",
            "intent_id": intent_id,
            "intent_name": intent_name,
            "expected_sub_agent": sub_agent,
            "difficulty": "concise",
            "utterance": f"Help with {intent_name.lower()} — {sample}",
        },
    ]
    return cases


def main():
    taxonomy = json.loads(TAXONOMY_PATH.read_text())
    all_cases = []
    for item in taxonomy["intents"]:
        all_cases.extend(generate_variations(item))

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in all_cases:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(all_cases)} evaluation test cases (3 per intent x 160 intents) to {OUT_PATH}")


if __name__ == "__main__":
    main()
