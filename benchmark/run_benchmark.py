#!/usr/bin/env python3
"""Runs intent-routing benchmark against CXAS (GECX) Apps (V1 LLM Router and/or V2 Fast Intent Router)
using exact server-side telemetry from CES diagnosticInfo.rootSpan, with HTTP 429 backoff & resume."""

import argparse
import concurrent.futures
import json
import os
import shutil
import statistics
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
APP_IDS_PATH = BASE_DIR.parent / "cxas_config" / "app_ids.json"
DATASET_PATH = BASE_DIR / "telco_eval_dataset.jsonl"
RESULTS_DIR = BASE_DIR / "results"

# Configurable LLM token pricing per 1M tokens (USD) and root agent name
LLM_INPUT_COST_PER_1M = float(os.environ.get("LLM_INPUT_COST_PER_1M", "0.15"))
LLM_OUTPUT_COST_PER_1M = float(os.environ.get("LLM_OUTPUT_COST_PER_1M", "0.60"))
ROOT_AGENT_DISPLAY_NAME = os.environ.get("CXAS_ROOT_AGENT_NAME", "Telecommunications Steering agent")
DEFAULT_LLM_MODEL_NAME = os.environ.get("BASELINE_LLM_MODEL_NAME", "default-llm")


def get_token() -> str:
    env_tok = os.environ.get("GOOGLE_ACCESS_TOKEN")
    if env_tok:
        return env_tok.strip()
    gcloud_bin = shutil.which("gcloud") or "gcloud"
    return subprocess.check_output([gcloud_bin, "auth", "print-access-token"]).decode().strip()


def parse_duration_ms(dur_str: str | None) -> float:
    if not dur_str or not isinstance(dur_str, str):
        return 0.0
    if dur_str.endswith("s"):
        try:
            return round(float(dur_str[:-1]) * 1000.0, 2)
        except ValueError:
            return 0.0
    return 0.0


def run_single_session(
    app_name: str,
    project_id: str,
    token: str,
    utterance: str,
    agent_id_to_display_name: dict[str, str],
    max_retries: int = 8,
) -> dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "x-goog-user-project": project_id,
        "Content-Type": "application/json",
    }

    retries_429 = 0
    resp_json = {}
    error = None
    client_latency_ms = 0.0
    session_id = ""

    for attempt in range(max_retries + 1):
        session_id = f"bench-{uuid.uuid4()}"
        session_name = f"{app_name}/sessions/{session_id}"
        url = f"https://ces.googleapis.com/v1beta/{session_name}:runSession"
        payload = {
            "config": {"session": session_name},
            "inputs": [{"text": utterance}],
        }

        t0 = time.perf_counter()
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw_body = resp.read().decode("utf-8")
                client_latency_ms = (time.perf_counter() - t0) * 1000.0
                resp_json = json.loads(raw_body)
                error = None
                break
        except urllib.error.HTTPError as e:
            client_latency_ms = (time.perf_counter() - t0) * 1000.0
            err_text = e.read().decode("utf-8", errors="replace")
            if e.code == 429 and attempt < max_retries:
                retries_429 += 1
                backoff = min(30.0, 4.0 * (1.5 ** attempt))
                time.sleep(backoff)
                continue
            resp_json = {"http_error": e.code, "body": err_text}
            error = f"HTTP {e.code}: {err_text[:300]}"
            break
        except Exception as e:
            client_latency_ms = (time.perf_counter() - t0) * 1000.0
            resp_json = {}
            error = str(e)
            break

    routed_sub_agent = None
    predicted_intent_id = None
    intent_confidence = None
    sub_agent_confidence = None
    diffusion_gemma_inference_ms = 0.0
    response_texts = []
    tool_calls = []

    perceived_latency_ms = 0.0
    orchestrator_latency_ms = 0.0
    orchestrator_prefill_ms = 0.0
    orchestrator_ttfc_ms = 0.0
    sub_agent_latency_ms = 0.0
    tool_latency_ms = 0.0

    orchestrator_input_tokens = 0
    orchestrator_output_tokens = 0
    sub_agent_input_tokens = 0
    sub_agent_output_tokens = 0
    model_name = None

    for out in resp_json.get("outputs", []):
        if "text" in out and out["text"]:
            response_texts.append(out["text"])
        if "diagnosticInfo" in out:
            diag = out["diagnosticInfo"]
            for msg in diag.get("messages", []):
                for chunk in msg.get("chunks", []):
                    if "agentTransfer" in chunk:
                        xfer = chunk["agentTransfer"]
                        routed_sub_agent = xfer.get("displayName") or agent_id_to_display_name.get(
                            xfer.get("targetAgent", "")
                        )
                    if "toolCall" in chunk:
                        tool_calls.append(chunk["toolCall"])
                    if "toolResponse" in chunk:
                        tresp = chunk["toolResponse"].get("response", {})
                        if isinstance(tresp, dict):
                            if tresp.get("intent_id"):
                                predicted_intent_id = tresp.get("intent_id")
                            if tresp.get("intent_confidence") is not None:
                                intent_confidence = float(tresp.get("intent_confidence"))
                            if tresp.get("sub_agent_confidence") is not None:
                                sub_agent_confidence = float(tresp.get("sub_agent_confidence"))
                            if tresp.get("inference_latency_ms"):
                                diffusion_gemma_inference_ms = float(tresp.get("inference_latency_ms"))
                            if tresp.get("sub_agent") and not routed_sub_agent:
                                routed_sub_agent = tresp.get("sub_agent")

            root_span = diag.get("rootSpan", {})
            root_attrs = root_span.get("attributes", {})
            perceived_latency_ms = float(
                root_attrs.get("perceived latency (ms)")
                or parse_duration_ms(root_span.get("duration"))
                or client_latency_ms
            )

            for cspan in root_span.get("childSpans", []):
                sname = cspan.get("name", "")
                attrs = cspan.get("attributes", {})
                dur_ms = parse_duration_ms(cspan.get("duration"))
                agent_attr = attrs.get("agent", "")

                if sname == "LLM":
                    if not model_name and attrs.get("model"):
                        model_name = attrs.get("model")
                    in_tok = int(attrs.get("input token count", 0))
                    out_tok = int(attrs.get("output token count", 0))
                    if agent_attr == ROOT_AGENT_DISPLAY_NAME:
                        orchestrator_latency_ms += dur_ms
                        orchestrator_prefill_ms += float(attrs.get("[debug] prefill time (ms)", 0))
                        orchestrator_ttfc_ms += float(attrs.get("time to first chunk (ms)", 0))
                        orchestrator_input_tokens += in_tok
                        orchestrator_output_tokens += out_tok
                    else:
                        sub_agent_latency_ms += dur_ms
                        sub_agent_input_tokens += in_tok
                        sub_agent_output_tokens += out_tok
                elif sname == "Tool":
                    t_name = attrs.get("name", "")
                    if t_name == "transfer_to_agent":
                        arg_agent = attrs.get("args", {}).get("agent_name")
                        if arg_agent and not routed_sub_agent:
                            routed_sub_agent = arg_agent
                    else:
                        tool_latency_ms += dur_ms

    return {
        "session_id": session_id,
        "routed_sub_agent": routed_sub_agent,
        "predicted_intent_id": predicted_intent_id,
        "intent_confidence": intent_confidence,
        "sub_agent_confidence": sub_agent_confidence,
        "diffusion_gemma_inference_ms": round(diffusion_gemma_inference_ms, 2),
        "response_text": " | ".join(response_texts),
        "model": model_name,
        "client_latency_ms": round(client_latency_ms, 2),
        "perceived_latency_ms": round(perceived_latency_ms, 2),
        "orchestrator_latency_ms": round(orchestrator_latency_ms, 2),
        "orchestrator_prefill_ms": round(orchestrator_prefill_ms, 2),
        "orchestrator_ttfc_ms": round(orchestrator_ttfc_ms, 2),
        "tool_latency_ms": round(tool_latency_ms, 2),
        "sub_agent_latency_ms": round(sub_agent_latency_ms, 2),
        "orchestrator_input_tokens": orchestrator_input_tokens,
        "orchestrator_output_tokens": orchestrator_output_tokens,
        "sub_agent_input_tokens": sub_agent_input_tokens,
        "sub_agent_output_tokens": sub_agent_output_tokens,
        "total_input_tokens": orchestrator_input_tokens + sub_agent_input_tokens,
        "total_output_tokens": orchestrator_output_tokens + sub_agent_output_tokens,
        "retries_429": retries_429,
        "tool_calls": tool_calls,
        "error": error,
    }


def evaluate_case(case: dict, app_cfg: dict, project_id: str, token: str, id_to_name: dict, min_interval_s: float = 0.0) -> dict:
    if min_interval_s > 0:
        time.sleep(min_interval_s)
    res = run_single_session(
        app_name=app_cfg["app_name"],
        project_id=project_id,
        token=token,
        utterance=case["utterance"],
        agent_id_to_display_name=id_to_name,
    )
    expected = case["expected_sub_agent"]
    predicted = res["routed_sub_agent"]
    is_correct = (predicted == expected)
    intent_correct = (
        res["predicted_intent_id"] == case["intent_id"]
        if res["predicted_intent_id"]
        else None
    )

    orch_cost_usd = (
        (res["orchestrator_input_tokens"] / 1_000_000.0) * LLM_INPUT_COST_PER_1M
        + (res["orchestrator_output_tokens"] / 1_000_000.0) * LLM_OUTPUT_COST_PER_1M
    )
    total_cost_usd = (
        (res["total_input_tokens"] / 1_000_000.0) * LLM_INPUT_COST_PER_1M
        + (res["total_output_tokens"] / 1_000_000.0) * LLM_OUTPUT_COST_PER_1M
    )

    return {
        "case_id": case["case_id"],
        "intent_id": case["intent_id"],
        "intent_name": case["intent_name"],
        "difficulty": case["difficulty"],
        "utterance": case["utterance"],
        "expected_sub_agent": expected,
        "predicted_sub_agent": predicted,
        "predicted_intent_id": res["predicted_intent_id"],
        "intent_confidence": res["intent_confidence"],
        "sub_agent_confidence": res["sub_agent_confidence"],
        "correct": is_correct,
        "intent_correct": intent_correct,
        "model": res["model"],
        "client_latency_ms": res["client_latency_ms"],
        "perceived_latency_ms": res["perceived_latency_ms"],
        "orchestrator_latency_ms": res["orchestrator_latency_ms"],
        "orchestrator_prefill_ms": res["orchestrator_prefill_ms"],
        "orchestrator_ttfc_ms": res["orchestrator_ttfc_ms"],
        "tool_latency_ms": res["tool_latency_ms"],
        "diffusion_gemma_inference_ms": res["diffusion_gemma_inference_ms"],
        "sub_agent_latency_ms": res["sub_agent_latency_ms"],
        "orchestrator_input_tokens": res["orchestrator_input_tokens"],
        "orchestrator_output_tokens": res["orchestrator_output_tokens"],
        "sub_agent_input_tokens": res["sub_agent_input_tokens"],
        "sub_agent_output_tokens": res["sub_agent_output_tokens"],
        "total_input_tokens": res["total_input_tokens"],
        "total_output_tokens": res["total_output_tokens"],
        "orchestrator_cost_usd": round(orch_cost_usd, 6),
        "total_cost_usd": round(total_cost_usd, 6),
        "retries_429": res["retries_429"],
        "response_text": res["response_text"],
        "error": res["error"],
    }


def percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def summarize_dist(vals: list[float]) -> dict:
    s = sorted(vals)
    if not s:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(statistics.mean(s), 2),
        "p50": round(percentile(s, 50), 2),
        "p90": round(percentile(s, 90), 2),
        "p95": round(percentile(s, 95), 2),
        "p99": round(percentile(s, 99), 2),
        "min": round(min(s), 2),
        "max": round(max(s), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--difficulty", choices=["standard", "conversational", "concise", "all"], default="standard")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of test cases (0 = all matching)")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent workers")
    parser.add_argument("--delay", type=float, default=0.8, help="Per-worker pacing delay in seconds")
    parser.add_argument("--resume", action="store_true", help="Reuse non-error cases from existing results file")
    parser.add_argument("--probe", action="store_true", help="Run 1 probe call and print parsed telemetry")
    args = parser.parse_args()

    app_ids = json.loads(APP_IDS_PATH.read_text())
    project_id = app_ids["project_id"]
    app_key = "v1_llm_router" if args.version == "v1" else "v2_diffusion_gemma_router"
    app_cfg = app_ids[app_key]
    id_to_name = {v: k for k, v in app_cfg["sub_agents"].items()}

    cases = [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()]
    if args.difficulty != "all":
        cases = [c for c in cases if c["difficulty"] == args.difficulty]
    if args.limit > 0:
        cases = cases[: args.limit]

    token = get_token()

    if args.probe:
        probe_case = cases[0]
        res = evaluate_case(probe_case, app_cfg, project_id, token, id_to_name)
        print(json.dumps(res, indent=2))
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_jsonl = RESULTS_DIR / f"{args.version}_{args.difficulty}_results.jsonl"

    completed_by_id = {}
    if args.resume and out_jsonl.exists():
        for line in out_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("error") and row.get("predicted_sub_agent"):
                completed_by_id[row["case_id"]] = row
        print(f"Resuming: loaded {len(completed_by_id)} already-completed non-error cases from {out_jsonl.name}")

    pending_cases = [c for c in cases if c["case_id"] not in completed_by_id]
    results = list(completed_by_id.values())

    print(
        f"Running {len(pending_cases)} pending test cases (of {len(cases)} total) on "
        f"{app_cfg['display_name']} (workers={args.workers}, delay={args.delay}s)..."
    )

    t_start = time.perf_counter()
    if pending_cases:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_to_case = {
                pool.submit(evaluate_case, c, app_cfg, project_id, token, id_to_name, args.delay): c
                for c in pending_cases
            }
            for idx, fut in enumerate(concurrent.futures.as_completed(future_to_case), start=1):
                r = fut.result()
                results.append(r)
                if idx % 10 == 0 or idx == len(pending_cases):
                    valid_so_far = [x for x in results if not x["error"]]
                    acc_so_far = (
                        sum(1 for x in valid_so_far if x["correct"]) / len(valid_so_far) * 100.0
                        if valid_so_far
                        else 0.0
                    )
                    dg_lats = [x["diffusion_gemma_inference_ms"] for x in valid_so_far if x.get("diffusion_gemma_inference_ms", 0) > 0]
                    dg_p50 = percentile(sorted(dg_lats), 50) if dg_lats else 0.0
                    print(
                        f"  [{len(results)}/{len(cases)}] Sub-Agent Acc: {acc_so_far:.1f}% "
                        f"({len(valid_so_far)} valid) | Last: {r['case_id']} -> {r.get('predicted_intent_id')}/{r['predicted_sub_agent']} "
                        f"| DG GPU P50: {dg_p50:.1f} ms | Tool Span: {r['tool_latency_ms']} ms"
                    )
                    results.sort(key=lambda x: x["case_id"])
                    with out_jsonl.open("w", encoding="utf-8") as f:
                        for row in results:
                            f.write(json.dumps(row) + "\n")

    wall_time_s = time.perf_counter() - t_start
    results.sort(key=lambda x: x["case_id"])

    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    valid = [r for r in results if not r["error"]]
    total = len(results)
    correct = sum(1 for r in valid if r["correct"])
    intent_scored = [r for r in valid if r.get("intent_correct") is not None]
    intent_correct_cnt = sum(1 for r in intent_scored if r["intent_correct"])
    errors = sum(1 for r in results if r["error"])

    per_sub_agent = {}
    confusion = {}
    for r in results:
        exp = r["expected_sub_agent"]
        pred = r["predicted_sub_agent"] or "NONE"
        confusion.setdefault(exp, {})
        confusion[exp][pred] = confusion[exp].get(pred, 0) + 1

    for sub_name in app_cfg["sub_agents"].keys():
        sub_cases = [r for r in valid if r["expected_sub_agent"] == sub_name]
        sub_corr = sum(1 for r in sub_cases if r["correct"])
        sub_int_corr = sum(1 for r in sub_cases if r.get("intent_correct"))
        sub_orch_lats = [r["orchestrator_latency_ms"] for r in sub_cases if r["orchestrator_latency_ms"] > 0]
        sub_dg_lats = [r["diffusion_gemma_inference_ms"] for r in sub_cases if r.get("diffusion_gemma_inference_ms", 0) > 0]
        sub_tool_lats = [r["tool_latency_ms"] for r in sub_cases if r.get("tool_latency_ms", 0) > 0]
        sub_e2e_lats = [r["perceived_latency_ms"] for r in sub_cases if r["perceived_latency_ms"] > 0]
        per_sub_agent[sub_name] = {
            "total": len(sub_cases),
            "correct": sub_corr,
            "accuracy_pct": round((sub_corr / len(sub_cases) * 100.0), 2) if sub_cases else 0.0,
            "intent_160_accuracy_pct": round((sub_int_corr / len(sub_cases) * 100.0), 2) if sub_cases and intent_scored else None,
            "diffusion_gemma_gpu_p50_ms": round(percentile(sorted(sub_dg_lats), 50), 2) if sub_dg_lats else 0.0,
            "tool_span_p50_ms": round(percentile(sorted(sub_tool_lats), 50), 2) if sub_tool_lats else 0.0,
            "orchestrator_p50_ms": round(percentile(sorted(sub_orch_lats), 50), 2),
            "perceived_e2e_p50_ms": round(percentile(sorted(sub_e2e_lats), 50), 2),
        }

    avg_orch_cost = statistics.mean(r["orchestrator_cost_usd"] for r in valid) if valid else 0.0
    avg_total_cost = statistics.mean(r["total_cost_usd"] for r in valid) if valid else 0.0

    summary = {
        "version": args.version,
        "app_display_name": app_cfg["display_name"],
        "app_name": app_cfg["app_name"],
        "model": valid[0]["model"] if valid else DEFAULT_LLM_MODEL_NAME,
        "difficulty_filter": args.difficulty,
        "total_cases": total,
        "valid_cases": len(valid),
        "correct_cases": correct,
        "error_cases": errors,
        "routing_accuracy_pct": round(correct / len(valid) * 100.0, 2) if valid else 0.0,
        "fine_grained_160_intent_accuracy_pct": (
            round(intent_correct_cnt / len(intent_scored) * 100.0, 2) if intent_scored else None
        ),
        "diffusion_gemma_gpu_inference_latency_ms": summarize_dist(
            [r["diffusion_gemma_inference_ms"] for r in valid if r.get("diffusion_gemma_inference_ms", 0) > 0]
        ),
        "cloud_run_tool_span_latency_ms": summarize_dist(
            [r["tool_latency_ms"] for r in valid if r.get("tool_latency_ms", 0) > 0]
        ),
        "orchestrator_routing_latency_ms": summarize_dist(
            [r["orchestrator_latency_ms"] for r in valid if r["orchestrator_latency_ms"] > 0]
        ),
        "orchestrator_prefill_latency_ms": summarize_dist(
            [r["orchestrator_prefill_ms"] for r in valid if r["orchestrator_prefill_ms"] > 0]
        ),
        "perceived_e2e_latency_ms": summarize_dist(
            [r["perceived_latency_ms"] for r in valid if r["perceived_latency_ms"] > 0]
        ),
        "client_e2e_latency_ms": summarize_dist([r["client_latency_ms"] for r in valid]),
        "token_and_cost_metrics": {
            "avg_orchestrator_input_tokens": round(
                statistics.mean(r["orchestrator_input_tokens"] for r in valid), 1
            )
            if valid
            else 0,
            "avg_orchestrator_output_tokens": round(
                statistics.mean(r["orchestrator_output_tokens"] for r in valid), 1
            )
            if valid
            else 0,
            "avg_sub_agent_input_tokens": round(
                statistics.mean(r["sub_agent_input_tokens"] for r in valid), 1
            )
            if valid
            else 0,
            "avg_sub_agent_output_tokens": round(
                statistics.mean(r["sub_agent_output_tokens"] for r in valid), 1
            )
            if valid
            else 0,
            "avg_total_input_tokens": round(statistics.mean(r["total_input_tokens"] for r in valid), 1)
            if valid
            else 0,
            "avg_total_output_tokens": round(
                statistics.mean(r["total_output_tokens"] for r in valid), 1
            )
            if valid
            else 0,
            "orchestrator_routing_cost_per_1k_turns_usd": round(avg_orch_cost * 1000.0, 4),
            "total_turn_cost_per_1k_turns_usd": round(avg_total_cost * 1000.0, 4),
        },
        "per_sub_agent": per_sub_agent,
        "confusion_matrix": confusion,
        "wall_time_seconds": round(wall_time_s, 2),
    }

    out_summary = RESULTS_DIR / f"{args.version}_{args.difficulty}_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    print("\n=== BENCHMARK SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
