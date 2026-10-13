#!/usr/bin/env python3
"""Updates CXAS App V1 with instructions.txt (160 intents) and standardized sub-agents."""

import json
import os
import uuid
from pathlib import Path
from export_and_clone import (
    API_ROOT,
    BASE_DIR,
    LOCATION,
    PROJECT_ID,
    V1_APP_ID,
    api_request,
    get_standardized_subagent_instruction,
    get_token,
)


def main():
    token = get_token()
    v1_base = f"{API_ROOT}/apps/{V1_APP_ID}"

    # 1. Create a backup version snapshot in CXAS before modifying V1
    try:
        ver_payload = {
            "displayName": "Pre-160-intents-backup",
            "description": "Snapshot taken automatically before applying 160-intent instructions.txt",
        }
        ver_resp = api_request("POST", f"{v1_base}/versions", token, ver_payload)
        print(f"Created V1 version snapshot: {ver_resp.get('name', ver_resp)}")
    except Exception as e:
        print(f"Note on version snapshot: {e}")

    # 2. Fetch current agents in V1
    agents_resp = api_request("GET", f"{v1_base}/agents?pageSize=100", token)
    app_data = api_request("GET", v1_base, token)
    root_name = app_data["rootAgent"]

    # 3. Standardize all 8 sub-agents (ensures identical behavior to V2)
    for agent in agents_resp.get("agents", []):
        if agent["name"] == root_name:
            continue
        dname = agent["displayName"]
        desc = agent.get("description", f"Sub-agent for {dname}")
        clean_instr = get_standardized_subagent_instruction(dname, desc)
        patched = api_request(
            "PATCH",
            f"https://ces.googleapis.com/v1beta/{agent['name']}?updateMask=instruction",
            token,
            {"displayName": dname, "instruction": clean_instr},
        )
        print(f"  Standardized V1 sub-agent '{dname}': {patched['name']}")

    # 4. Update V1 Root Orchestrator with taxonomy/instructions.txt (160 intents)
    instr_path = BASE_DIR.parent / "taxonomy" / "instructions.txt"
    v1_instructions = instr_path.read_text()
    root_display_name = os.environ.get("CXAS_ROOT_AGENT_NAME", "Telecommunications Steering agent")

    patched_root = api_request(
        "PATCH",
        f"https://ces.googleapis.com/v1beta/{root_name}?updateMask=instruction",
        token,
        {
            "displayName": root_display_name,
            "instruction": v1_instructions,
        },
    )
    print(
        f"Updated V1 Root Orchestrator ({patched_root['name']}) with instructions.txt "
        f"({len(v1_instructions)} chars)"
    )


if __name__ == "__main__":
    main()
