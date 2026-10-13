# Telco 160-Intent Routing Benchmark: LLM Router vs. DiffusionGemma

This repository benchmarks two intent-routing architectures for a **160-intent Telecommunications Customer Experience Agent** across **8 specialist sub-agents** (`Accounts`, `Billing`, `Payments`, `Disconnect`, `Plans`, `Equipment`, `Usage`, and `Technical Support`) in Gemini Enterprise for Customer Experience (GECX / CXAS):

1. **Version 1 (Baseline LLM Router)**: All 160 intent definitions, descriptions, sample utterances, and disambiguation rules (~13,800 tokens) are placed directly inside the Root Orchestrator's [`taxonomy/instructions.txt`](taxonomy/instructions.txt).
2. **Version 2 (Fast System-1 Intent Router)**: The Root Orchestrator uses a compact prompt ([`taxonomy/instructions_v2.txt`](taxonomy/instructions_v2.txt)) and delegates 160-intent classification to **DiffusionGemma** served on a Vertex AI Dedicated Endpoint (`vLLM` with automatic prefix caching) behind a lightweight Cloud Run OpenAPI tool ([`diffusion_gemma_service/app.py`](diffusion_gemma_service/app.py)).

---

## Repository Structure

```text
├── .env.example                          # Environment variable template (no hardcoded project IDs)
├── taxonomy/
│   ├── generate_taxonomy.py              # Generates the 160 Telco intents and CXAS instruction files
│   ├── telco_160_intents.json            # Canonical 160-intent catalog (20 intents x 8 sub-agents)
│   ├── instructions.txt                  # Version 1 Root Orchestrator prompt (160 intents inline)
│   └── instructions_v2.txt               # Version 2 Root Orchestrator prompt (tool-delegated routing)
├── diffusion_gemma_service/
│   ├── app.py                            # Cloud Run OpenAPI classifier wrapper + logprobs confidence parser
│   ├── Dockerfile                        # Non-root container image for Cloud Run deployment
│   ├── openapi_spec.json                 # OpenAPI 3.0 specification registered as a CXAS Toolset
│   └── telco_160_intents.json            # Bundled 160-intent taxonomy for prefix-cached system prompt
├── cxas_agents/
│   ├── v1_llm_router/                    # Complete CXAS App V1 agent definitions & instructions (9 agents)
│   └── v2_fast_intent_router/            # Complete CXAS App V2 agent definitions, OpenAPI toolset & instructions
├── cxas_config/
│   ├── export_and_clone.py               # Exports V1 scaffolding and clones V2 CXAS app via CES v1beta API
│   ├── update_v1.py                      # Snapshots and updates V1 with the 160-intent instructions.txt
│   └── configure_v2_tool.py              # Registers the Cloud Run OpenAPI Toolset on V2
├── benchmark/
│   ├── generate_eval_dataset.py          # Generates 480 stratified test utterances (3 per intent)
│   ├── telco_eval_dataset.jsonl          # Evaluation dataset
│   ├── run_benchmark.py                  # Runs live CXAS sessions and extracts rootSpan telemetry
│   ├── compare_versions.py               # Compares V1 and V2 summary JSON metrics
│   └── results/                          # Sanitized benchmark summary & per-utterance JSONL outputs
└── reports/
    ├── experiment_report.md              # Side-by-side KPI comparison table
    └── v1_baseline_report.md             # Version 1 baseline evaluation report
```

---

## Quick Start

1. **Configure Environment Variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your GCP_PROJECT_ID, CXAS_V1_APP_ID, and ROUTER_ENDPOINT_DNS / PATH
   export $(grep -v '^#' .env | xargs)
   ```

2. **Deploy the Cloud Run Intent Classifier Service**:
   ```bash
   gcloud run deploy intent-router-service \
     --source ./diffusion_gemma_service \
     --project "${GCP_PROJECT_ID}" \
     --region "${GCP_REGION:-asia-southeast1}" \
     --set-env-vars "ROUTER_ENDPOINT_DNS=${ROUTER_ENDPOINT_DNS},ROUTER_ENDPOINT_PATH=${ROUTER_ENDPOINT_PATH},ROUTER_MODEL_NAME=${ROUTER_MODEL_NAME}" \
     --min-instances 1 \
     --memory 512Mi
   ```

3. **Configure CXAS Apps (V1 & V2)**:
   ```bash
   python3 cxas_config/export_and_clone.py
   python3 cxas_config/update_v1.py
   python3 cxas_config/configure_v2_tool.py --url "${CLOUD_RUN_SERVICE_URL}"
   ```

4. **Run the 160-Utterance Benchmark**:
   ```bash
   python3 benchmark/run_benchmark.py --version v1 --difficulty standard --workers 2 --delay 0.8
   python3 benchmark/run_benchmark.py --version v2 --difficulty standard --workers 2 --delay 0.8
   ```
