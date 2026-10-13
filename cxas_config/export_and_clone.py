#!/usr/bin/env python3
"""Exports the CXAS scaffolding app (V1) and creates a clean cloned CXAS app (V2) for the intent routing experiment."""

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "<YOUR_GCP_PROJECT_ID>")
LOCATION = os.environ.get("GCP_LOCATION", "us")
V1_APP_ID = os.environ.get("CXAS_V1_APP_ID", "<YOUR_V1_APP_ID>")
V2_DISPLAY_NAME = os.environ.get("CXAS_V2_APP_NAME", "telco-agent-fast-intent-router")
API_ROOT = f"https://ces.googleapis.com/v1beta/projects/{PROJECT_ID}/locations/{LOCATION}"


def get_token() -> str:
    env_tok = os.environ.get("GOOGLE_ACCESS_TOKEN")
    if env_tok:
        return env_tok.strip()
    gcloud_bin = shutil.which("gcloud") or "gcloud"
    return subprocess.check_output([gcloud_bin, "auth", "print-access-token"]).decode().strip()


def api_request(method: str, url: str, token: str, payload: dict | None = None) -> dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "x-goog-user-project": PROJECT_ID,
        "Content-Type": "application/json",
    }
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} on {method} {url}: {err_body}") from e


def export_scaffolding(token: str) -> dict:
    v1_base = f"{API_ROOT}/apps/{V1_APP_ID}"
    app_data = api_request("GET", v1_base, token)
    agents_data = api_request("GET", f"{v1_base}/agents?pageSize=100", token)
    tools_data = api_request("GET", f"{v1_base}/tools?pageSize=100", token)

    bundle = {
        "app": app_data,
        "agents": agents_data.get("agents", []),
        "tools": tools_data.get("tools", []),
    }
    export_path = BASE_DIR / "scaffolding_export.json"
    export_path.write_text(json.dumps(bundle, indent=2))
    print(f"Saved scaffolding backup to {export_path} ({len(bundle['agents'])} agents)")
    return bundle


def get_standardized_subagent_instruction(display_name: str, description: str) -> str:
    """Returns a clean, consistent sub-agent instruction without the scaffolding copy-paste bug."""
    return (
        f"<role>You are an Agent that helps the user with {display_name} related inquiries.</role>\n"
        f"<persona>\n"
        f"    Your primary goal is to assist the user with telecommunications {display_name} requests.\n"
        f"</persona>\n"
        f"<constraints>\n"
        f"    Strict to routing action, do not provide any further information or involve in any other topic discussions.\n"
        f"</constraints>\n"
        f"<taskflow>\n"
        f"    These define the conversational subtasks that you can take. Each\n"
        f"    subtask has a sequence of steps that should be taken in order.\n"
        f"    <subtask name=\"{display_name}\">\n"
        f"        <step name=\"{display_name}\">\n"
        f"            <trigger>User query is routed to {display_name} ({description})</trigger>\n"
        f"            <action>Inform the user that you can better assist them with this {display_name} request.</action>\n"
        f"        </step>\n"
        f"    </subtask>\n"
        f"</taskflow>\n"
        f"<examples>\n"
        f"</examples>"
    )


def create_or_get_v2_app(token: str, scaffolding: dict) -> dict:
    v2_display_name = V2_DISPLAY_NAME
    apps_resp = api_request("GET", f"{API_ROOT}/apps?pageSize=100", token)
    existing_app = None
    for app in apps_resp.get("apps", []):
        if app.get("displayName") == v2_display_name:
            existing_app = app
            print(f"Found existing V2 app: {app['name']}")
            break

    if not existing_app:
        v1_app = scaffolding["app"]
        create_payload = {
            "displayName": v2_display_name,
            "description": "Experiment Version 2: CXAS Telco Agent using DiffusionGemma on Cloud Run to classify 160 intents across 8 sub-agents.",
            "loggingSettings": v1_app.get("loggingSettings", {}),
            "timeZoneSettings": v1_app.get("timeZoneSettings", {"timeZone": "America/Los_Angeles"}),
            "variableDeclarations": v1_app.get("variableDeclarations", []),
        }
        op = api_request("POST", f"{API_ROOT}/apps", token, create_payload)
        print(f"Create App response: {json.dumps(op, indent=2)}")
        # Check if long-running operation or immediate resource
        if "response" in op and "name" in op["response"]:
            existing_app = op["response"]
        elif "name" in op and "/operations/" in op["name"]:
            import time
            op_name = op["name"]
            for _ in range(30):
                time.sleep(2)
                op_status = api_request("GET", f"https://ces.googleapis.com/v1beta/{op_name}", token)
                if op_status.get("done"):
                    if "error" in op_status:
                        raise RuntimeError(f"Operation failed: {op_status['error']}")
                    existing_app = op_status.get("response")
                    break
        else:
            existing_app = op

    v2_app_name = existing_app["name"]
    v2_app_id = v2_app_name.split("/")[-1]
    print(f"V2 App ready: {v2_app_name} (ID: {v2_app_id})")

    # Inspect existing agents in V2 app
    existing_agents_resp = api_request("GET", f"{API_ROOT}/apps/{v2_app_id}/agents?pageSize=100", token)
    existing_by_name = {a["displayName"]: a for a in existing_agents_resp.get("agents", [])}

    v1_root_name = scaffolding["app"]["rootAgent"]
    v1_sub_agents = [a for a in scaffolding["agents"] if a["name"] != v1_root_name]
    v1_root_agent = next(a for a in scaffolding["agents"] if a["name"] == v1_root_name)

    v2_child_agent_names = []
    v2_sub_agent_map = {}

    for sub in v1_sub_agents:
        dname = sub["displayName"]
        desc = sub.get("description", f"Sub-agent for {dname}")
        clean_instr = get_standardized_subagent_instruction(dname, desc)
        if dname in existing_by_name:
            agent_obj = existing_by_name[dname]
            print(f"  Sub-agent '{dname}' already exists: {agent_obj['name']}")
        else:
            payload = {
                "displayName": dname,
                "description": desc,
                "instruction": clean_instr,
                "tools": [
                    f"projects/{PROJECT_ID}/locations/{LOCATION}/apps/{v2_app_id}/tools/end_session"
                ],
            }
            agent_obj = api_request("POST", f"{API_ROOT}/apps/{v2_app_id}/agents", token, payload)
            print(f"  Created sub-agent '{dname}': {agent_obj['name']}")
        v2_child_agent_names.append(agent_obj["name"])
        v2_sub_agent_map[dname] = agent_obj["name"]

    # Create or update the Root Orchestrator Agent in V2
    v2_instr_path = BASE_DIR.parent / "taxonomy" / "instructions_v2.txt"
    v2_instr = v2_instr_path.read_text() if v2_instr_path.exists() else v1_root_agent.get("instruction", "")

    root_dname = v1_root_agent["displayName"]
    if root_dname in existing_by_name:
        root_obj = existing_by_name[root_dname]
        patch_payload = {
            "displayName": root_dname,
            "description": v1_root_agent.get("description", ""),
            "instruction": v2_instr,
            "childAgents": v2_child_agent_names,
            "tools": [
                f"projects/{PROJECT_ID}/locations/{LOCATION}/apps/{v2_app_id}/tools/end_session"
            ],
        }
        root_obj = api_request(
            "PATCH",
            f"https://ces.googleapis.com/v1beta/{root_obj['name']}?updateMask=instruction,childAgents,description,tools",
            token,
            patch_payload,
        )
        print(f"  Updated root agent '{root_dname}': {root_obj['name']}")
    else:
        create_root_payload = {
            "displayName": root_dname,
            "description": v1_root_agent.get("description", ""),
            "instruction": v2_instr,
            "childAgents": v2_child_agent_names,
            "tools": [
                f"projects/{PROJECT_ID}/locations/{LOCATION}/apps/{v2_app_id}/tools/end_session"
            ],
        }
        root_obj = api_request("POST", f"{API_ROOT}/apps/{v2_app_id}/agents", token, create_root_payload)
        print(f"  Created root agent '{root_dname}': {root_obj['name']}")

    # Set rootAgent on V2 App
    patched_app = api_request(
        "PATCH",
        f"{API_ROOT}/apps/{v2_app_id}?updateMask=rootAgent",
        token,
        {"rootAgent": root_obj["name"]},
    )
    print(f"Linked rootAgent on V2 App: {patched_app.get('rootAgent')}")

    app_ids_doc = {
        "project_id": PROJECT_ID,
        "location": LOCATION,
        "v1_llm_router": {
            "app_id": V1_APP_ID,
            "app_name": scaffolding["app"]["name"],
            "display_name": scaffolding["app"]["displayName"],
            "console_url": f"https://ces.cloud.google.com/projects/{PROJECT_ID}/locations/{LOCATION}/apps/{V1_APP_ID}",
            "root_agent": v1_root_name,
            "sub_agents": {a["displayName"]: a["name"] for a in v1_sub_agents},
        },
        "v2_diffusion_gemma_router": {
            "app_id": v2_app_id,
            "app_name": v2_app_name,
            "display_name": v2_display_name,
            "console_url": f"https://ces.cloud.google.com/projects/{PROJECT_ID}/locations/{LOCATION}/apps/{v2_app_id}",
            "root_agent": root_obj["name"],
            "sub_agents": v2_sub_agent_map,
        },
    }
    ids_path = BASE_DIR / "app_ids.json"
    ids_path.write_text(json.dumps(app_ids_doc, indent=2))
    print(f"Wrote app registry to {ids_path}")
    return app_ids_doc


if __name__ == "__main__":
    tok = get_token()
    scaff = export_scaffolding(tok)
    create_or_get_v2_app(tok, scaff)
