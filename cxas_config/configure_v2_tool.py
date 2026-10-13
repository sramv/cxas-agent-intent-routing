#!/usr/bin/env python3
"""Registers the Cloud Run OpenAPI Toolset on CXAS App V2 and links it to the V2 Root Orchestrator."""

import argparse
import json
import os
from pathlib import Path
from export_and_clone import API_ROOT, BASE_DIR, LOCATION, PROJECT_ID, api_request, get_token


def build_openapi_spec(cloud_run_url: str) -> str:
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Telco Intent Classifier API",
            "version": "1.0.0",
            "description": "Classifies user telecommunications queries into 1 of 160 intents and returns the target sub-agent.",
        },
        "servers": [{"url": cloud_run_url.rstrip("/")}],
        "paths": {
            "/classify": {
                "post": {
                    "operationId": "classify_telco_intent",
                    "summary": "Classify a customer telecommunications query into one of 160 intents and 8 sub-agents.",
                    "description": "Calls the intent router service on Cloud Run to classify a customer telco utterance into intent_id and sub_agent.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["utterance"],
                                    "properties": {
                                        "utterance": {
                                            "type": "string",
                                            "description": "The exact customer query to classify.",
                                        }
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Classification result with intent_id, target sub_agent, and confidence scores.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "intent_id": {"type": "string"},
                                            "intent_name": {"type": "string"},
                                            "sub_agent": {"type": "string"},
                                            "intent_confidence": {"type": "number"},
                                            "sub_agent_confidence": {"type": "number"},
                                            "inference_latency_ms": {"type": "number"},
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            }
        },
    }
    return json.dumps(spec, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default=os.environ.get("CLOUD_RUN_SERVICE_URL", ""),
        help="Deployed Cloud Run base URL (or set CLOUD_RUN_SERVICE_URL)",
    )
    args = parser.parse_args()
    if not args.url:
        raise ValueError("Please provide --url or set CLOUD_RUN_SERVICE_URL")

    token = get_token()
    app_ids = json.loads((BASE_DIR / "app_ids.json").read_text())
    v2_cfg = app_ids["v2_diffusion_gemma_router"]
    v2_app_id = v2_cfg["app_id"]
    v2_root_agent = v2_cfg["root_agent"]
    v2_base = f"{API_ROOT}/apps/{v2_app_id}"

    openapi_schema = build_openapi_spec(args.url)
    (BASE_DIR.parent / "diffusion_gemma_service" / "openapi_spec.json").write_text(openapi_schema)

    toolset_id = os.environ.get("CXAS_TOOLSET_ID", "diffusion_gemma_classifier")

    # Check if toolset already exists
    toolsets_resp = api_request("GET", f"{v2_base}/toolsets?pageSize=100", token)
    existing_ts = None
    for ts in toolsets_resp.get("toolsets", []):
        if ts.get("displayName") == toolset_id:
            existing_ts = ts
            break

    ts_payload = {
        "displayName": toolset_id,
        "description": "160-intent classifier hosted on Cloud Run",
        "openApiToolset": {
            "openApiSchema": openapi_schema,
        },
    }

    if existing_ts:
        ts_obj = api_request(
            "PATCH",
            f"https://ces.googleapis.com/v1beta/{existing_ts['name']}?updateMask=openApiToolset,description",
            token,
            ts_payload,
        )
        print(f"Updated V2 OpenAPI Toolset: {ts_obj['name']}")
    else:
        ts_obj = api_request("POST", f"{v2_base}/toolsets?toolsetId={toolset_id}", token, ts_payload)
        print(f"Created V2 OpenAPI Toolset: {ts_obj['name']}")

    # Attach toolset to V2 Root Orchestrator Agent
    v2_instr = (BASE_DIR.parent / "taxonomy" / "instructions_v2.txt").read_text()
    root_display_name = os.environ.get("CXAS_ROOT_AGENT_NAME", "Telecommunications Steering agent")
    patched_root = api_request(
        "PATCH",
        f"https://ces.googleapis.com/v1beta/{v2_root_agent}?updateMask=instruction,toolsets,tools",
        token,
        {
            "displayName": root_display_name,
            "instruction": v2_instr,
            "toolsets": [
                {
                    "toolset": ts_obj["name"],
                    "toolIds": ["classify_telco_intent"],
                }
            ],
            "tools": [
                f"projects/{PROJECT_ID}/locations/{LOCATION}/apps/{v2_app_id}/tools/end_session",
            ],
        },
    )
    print(f"Attached OpenAPI Toolset to V2 Root Agent: {patched_root['name']}")


if __name__ == "__main__":
    main()
