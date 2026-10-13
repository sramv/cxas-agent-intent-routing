# CXAS Telco Agent Routing Experiment — Baseline Results (Version 1: LLM Router)

See full summary JSON in `../benchmark/results/v1_standard_summary.json` and per-utterance telemetry in `../benchmark/results/v1_standard_results.jsonl`.

## Key Baseline Metrics (160 Intents across 8 Sub-Agents)
- **Model**: `baseline-llm`
- **Routing Accuracy**: `100.0%` (`160 / 160` across all 8 sub-agents: `Accounts`, `Billing`, `Payments`, `Disconnect`, `Plans`, `Equipment`, `Usage`, `Technical Support`)
- **Orchestrator Routing LLM Latency**:
  - **Mean**: `928.50 ms`
  - **$P_{50}$**: `891.92 ms` (Prefill $P_{50}$: `98.0 ms`)
  - **$P_{90}$**: `1,174.86 ms`
  - **$P_{95}$**: `1,254.31 ms`
  - **$P_{99}$**: `1,372.88 ms`
- **Server Perceived End-to-End Turn Latency**:
  - **Mean**: `1,821.53 ms`
  - **$P_{50}$**: `1,761.00 ms`
  - **$P_{90}$**: `2,105.00 ms`
  - **$P_{95}$**: `2,268.95 ms`
  - **$P_{99}$**: `2,644.30 ms`
- **Token & Cost Profile**:
  - **Orchestrator Input Tokens / Turn**: `13,794.9`
  - **Orchestrator Output Tokens / Turn**: `24.1`
  - **Sub-Agent Input / Output Tokens / Turn**: `1,469.1` / `16.9`
  - **Orchestrator Routing Cost / 1,000 Turns**: `$2.0837`
  - **Total Turn Cost / 1,000 Turns**: `$2.3142`
