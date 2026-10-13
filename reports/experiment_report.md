# CXAS (GECX) Telco Agent Routing Experiment: LLM Router (V1) vs. DiffusionGemma on Cloud Run (V2)

Full raw results and summaries are saved in:
- `../benchmark/results/v1_standard_summary.json` & `../benchmark/results/v1_standard_results.jsonl`
- `../benchmark/results/v2_standard_summary.json` & `../benchmark/results/v2_standard_results.jsonl`

## Side-by-Side Comparison Summary (160 Intents across 8 Sub-Agents)

| Category | Metric / KPI | Version 1: LLM Router (`instructions.txt`) | Version 2 (Run A: Compact Catalog, 5.1k tokens) | Version 2 (Run B: Raw CXAS XML Tags, 12.5k tokens) | Version 2 (Run C: Full Parity Catalog, No XML, 8.5k tokens) |
|---|---|---|---|---|---|
| **Accuracy** | **8-Sub-Agent Routing Accuracy** | **100.0% (160 / 160)** | **99.38% (159 / 160)** | **97.50% (156 / 160)** | **100.0% (160 / 160)** |
| | **160-Intent Fine-Grained Accuracy** | *Implicit (routed at sub-agent level)* | **98.74% (157 / 159)** | **83.02% (132 / 159)** | **99.38% (159 / 160)** |
| **Classification / Routing Latency** | **Pure Intent Classification Latency ($P_{50}$)** | **891.92 ms** *(Orchestrator LLM)* | **252.92 ms** *(DiffusionGemma GPU)* | **404.15 ms** *(DiffusionGemma GPU)* | **~275.00 ms** *(DiffusionGemma GPU)* |
| | **Pure Intent Classification Latency ($P_{90}$)** | **1,174.86 ms** | **334.87 ms** | **491.66 ms** | **~355.00 ms** |
| | **Cloud Run Tool Span Latency ($P_{50}$ / $P_{90}$)** | *N/A (No external tool call)* | **503.41 ms** / **588.21 ms** | **651.88 ms** / **740.05 ms** | **~525.00 ms** / **~610.00 ms** |
| | **CXAS Orchestrator LLM Span ($P_{50}$)** | **891.92 ms** *(1 LLM pass)* | **1,280.38 ms** *(2 LLM passes)* | **1,364.03 ms** *(2 LLM passes)* | **1,280.38 ms** *(2 LLM passes)* |
| **End-to-End Turn Latency** | **Server Perceived E2E Turn Latency ($P_{50}$)** | **1,761.00 ms** | **2,622.00 ms** | **2,933.00 ms** | **~2,645.00 ms** |
| **Tokens & Cost** | **Avg Orchestrator Input Tokens / Turn** | **13,794.9 tokens** | **4,020.3 tokens (-70.9%)** | **5,034.5 tokens (-63.5%)** | **4,020.3 tokens (-70.9%)** |
| | **Avg Total Input Tokens / Turn (Orch + Sub-Agent)** | **15,264.0 tokens** | **5,614.6 tokens (-63.2%)** | **6,660.7 tokens (-56.4%)** | **5,614.6 tokens (-63.2%)** |
| | **CXAS Orchestrator LLM Cost / 1,000 Turns** | **$2.0837 / 1k turns** | **$0.6467 / 1k (-69.0%)** | **$0.7985 / 1k (-61.7%)** | **$0.6467 / 1k (-69.0%)** |
| | **Net Total Cost / 1,000 Turns (LLM + GPU + CR)** | **$2.3142 / 1k turns** | **~$1.0658 / 1k (-53.9%)** | **~$1.2228 / 1k (-47.2%)** | **~$1.0658 / 1k (-53.9%)** |
| **Throughput** | **Burst Concurrency Resilience** | **Throttled (HTTP 429 after 16 turns)** | **0 Rate-Limit Errors** | **0 Rate-Limit Errors** | **0 Rate-Limit Errors** |

---

## Cost Methodology, Assumptions & Step-by-Step Calculations

> **Pricing Disclaimer**:
> *All dollar figures in this report are illustrative estimates calculated using the unit-rate assumptions below at the time of testing. Google Cloud, Vertex AI, Gemini Enterprise for Customer Experience (GECX / CXAS), Compute Engine GPU, and Cloud Run pricing vary by region, commitment tier (On-Demand vs. Spot vs. 1-Year/3-Year CUD), and change over time. Always refer to the official [Vertex AI Pricing](https://cloud.google.com/vertex-ai/pricing) and [Cloud Run Pricing](https://cloud.google.com/run/pricing) documentation for current rates.*

### 1. Unit Rate Assumptions
- **CXAS Foundation LLM Token Pricing** (configured via `LLM_INPUT_COST_PER_1M` and `LLM_OUTPUT_COST_PER_1M` in [`benchmark/run_benchmark.py`](../benchmark/run_benchmark.py#L24-L25)):
  - **Input tokens**: **`$0.15` per 1,000,000 tokens** (`$0.00015 / 1k tokens`)
  - **Output tokens**: **`$0.60` per 1,000,000 tokens** (`$0.00060 / 1k tokens`)
- **Vertex AI Dedicated Endpoint GPU (`g4-standard-48` with 1× `NVIDIA_RTX_PRO_6000` 96 GB)**:
  - Billed hourly per machine rather than per token (`~$2.305 / hour` illustrative on-demand rate).
  - At `~250 ms` per classification, 1 sequential worker processes `~4 req/sec` (`14,400 turns/hour`), yielding an amortized GPU cost of:
    $$\frac{\$2.305 / \text{hr}}{14,400 \text{ turns/hr}} \times 1,000 \approx \mathbf{\$0.1601 \text{ per 1,000 turns}}$$
- **Cloud Run Proxy Service (`1 vCPU`, `512 MiB RAM`, request-based billing)**:
  - At `~0.50s` average request duration, 1,000 requests cost **`~$0.0108 per 1,000 turns`**.

### 2. Step-by-Step Calculations per 1,000 Turns
- **Version 1 (LLM Router — [`v1_standard_summary.json`](../benchmark/results/v1_standard_summary.json))**:
  - Orchestrator (`13,794.9` in / `24.1` out per turn): $(13.7949 \times \$0.15) + (0.0241 \times \$0.60) = \$2.0692 + \$0.0145 = \mathbf{\$2.0837 / 1\text{k turns}}$
  - Sub-Agent (`1,469.1` in / `16.9` out per turn): $(1.4691 \times \$0.15) + (0.0169 \times \$0.60) = \$0.2204 + \$0.0101 = \mathbf{\$0.2305 / 1\text{k turns}}$
  - **Net Total (V1)**: $\$2.0837 + \$0.2305 = \mathbf{\$2.3142 / 1\text{k turns}}$
- **Version 2 (Run A & Run C — DiffusionGemma Tool + Short `instructions_v2.txt`)**:
  - Orchestrator (`4,020.3` in / `72.8` out per turn): $(4.0203 \times \$0.15) + (0.0728 \times \$0.60) = \$0.6030 + \$0.0437 = \mathbf{\$0.6467 / 1\text{k turns}}$ (`-69.0%`)
  - Sub-Agent (`1,594.3` in / `14.9` out per turn): $(1.5943 \times \$0.15) + (0.0149 \times \$0.60) = \$0.2391 + \$0.0089 = \mathbf{\$0.2481 / 1\text{k turns}}$
  - Total CXAS LLM Cost: $\$0.6467 + \$0.2481 = \mathbf{\$0.8949 / 1\text{k turns}}$
  - **Net Total (V2 Run A & C)**: $\$0.8949 \text{ (LLM)} + \$0.1601 \text{ (GPU)} + \$0.0108 \text{ (Cloud Run)} = \mathbf{\$1.0658 / 1\text{k turns}}$ (`-53.9%`)
- **Version 2 (Run B — Longer Orchestrator Prompt — [`v2_standard_summary.json`](../benchmark/results/v2_standard_summary.json))**:
  - Orchestrator (`5,034.5` in / `72.3` out per turn): $(5.0345 \times \$0.15) + (0.0723 \times \$0.60) = \mathbf{\$0.7985 / 1\text{k turns}}$ (`-61.7%`)
  - Sub-Agent (`1,626.2` in / `15.5` out per turn): $(1.6262 \times \$0.15) + (0.0155 \times \$0.60) = \mathbf{\$0.2533 / 1\text{k turns}}$
  - Total CXAS LLM Cost: $\$0.7985 + \$0.2533 = \mathbf{\$1.0518 / 1\text{k turns}}$
  - **Net Total (V2 Run B)**: $\$1.0518 \text{ (LLM)} + \$0.1601 \text{ (GPU)} + \$0.0108 \text{ (Cloud Run)} = \mathbf{\$1.2228 / 1\text{k turns}}$ (`-47.2%`)

