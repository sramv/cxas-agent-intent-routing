#!/usr/bin/env python3
"""Cloud Run microservice that classifies Telco utterances into 1 of 160 intents and 8 sub-agents
using a dedicated Vertex AI OpenAI-compatible chat completions endpoint."""

import json
import logging
import math
import os
import re
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("intent-router-service")

TAXONOMY_PATH = Path(__file__).resolve().parent / "telco_160_intents.json"
TAXONOMY = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))

# Build deterministic lookup tables for allow-list validation
INTENT_MAP: dict[str, dict] = {it["intent_id"]: it for it in TAXONOMY["intents"]}
VALID_SUB_AGENTS: set[str] = set(TAXONOMY["sub_agents"].keys())

# Vertex AI Dedicated Endpoint configuration via environment variables (no hardcoded project or endpoint IDs)
ENDPOINT_DNS = (
    os.environ.get("ROUTER_ENDPOINT_DNS")
    or os.environ.get("DIFFUSION_GEMMA_DNS")
    or "https://<YOUR_ENDPOINT_DNS>.prediction.vertexai.goog"
).rstrip("/")
ENDPOINT_PATH = (
    os.environ.get("ROUTER_ENDPOINT_PATH")
    or os.environ.get("DIFFUSION_GEMMA_ENDPOINT_PATH")
    or "projects/<YOUR_PROJECT_NUMBER>/locations/<YOUR_REGION>/endpoints/<YOUR_ENDPOINT_ID>"
).strip("/")
ROUTER_MODEL_NAME = os.environ.get("ROUTER_MODEL_NAME", "intent-classifier-model")
VERTEX_CHAT_URL = f"{ENDPOINT_DNS}/v1beta1/{ENDPOINT_PATH}/chat/completions"

# Build static XML-free pipe-delimited catalog including sample utterances (100% parity with baseline taxonomy).
# Keeping this free of orchestrator XML tags (<taskflow>, {@AGENT:...}) avoids output format interference,
# while vLLM automatic prefix caching pins all ~8,546 static tokens in GPU KV memory across requests.
_CATALOG_LINES = [
    f"{it['intent_id']}|{it['sub_agent']}|{it['intent_name']}: {it['description']} (Sample: \"{it['sample_utterance']}\")"
    for it in TAXONOMY["intents"]
]
SYSTEM_PROMPT = (
    "You are a Telco Intent Classifier. Classify the user query into one of the 160 intents below.\n"
    'Reply ONLY with valid JSON: {"intent_id": "<ID>", "sub_agent": "<SubAgent>"}\n\n'
    "INTENT CATALOG (intent_id|sub_agent|name: description (Sample)):\n" + "\n".join(_CATALOG_LINES)
)

# Cached GCP access token (refreshed automatically via Cloud Run Metadata Server)
_TOKEN_CACHE = {"token": None, "expires_at": 0.0}
_INTENT_ID_REGEX = re.compile(r"\b(ACC|BIL|PAY|DSC|PLN|EQP|USG|TEC)_(\d{3})\b")
_PREFIX_SET = {"ACC", "BIL", "PAY", "DSC", "PLN", "EQP", "USG", "TEC"}


def get_access_token() -> str:
    now = time.time()
    if _TOKEN_CACHE["token"] and now < _TOKEN_CACHE["expires_at"] - 60:
        return _TOKEN_CACHE["token"]

    # 1. Try Cloud Run / GCE Metadata Server
    meta_url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
    req = urllib.request.Request(meta_url, headers={"Metadata-Flavor": "Google"})
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            _TOKEN_CACHE["token"] = data["access_token"]
            _TOKEN_CACHE["expires_at"] = now + float(data.get("expires_in", 3000))
            return _TOKEN_CACHE["token"]
    except Exception:
        pass

    # 2. Fallback for local testing via environment variable
    env_tok = os.environ.get("GOOGLE_ACCESS_TOKEN")
    if env_tok:
        return env_tok

    raise RuntimeError("Unable to acquire GCP access token from metadata server or GOOGLE_ACCESS_TOKEN")


def extract_confidence_metrics(choice: dict) -> dict:
    """Computes exact joint probability P(intent_id) and P(sub_agent) from vLLM token logprobs.
    Handles both single-token prefixes (ACC, BIL, PAY, DSC, TEC) and two-token SentencePiece prefixes (PLN, EQP, USG)."""
    lp_items = (choice.get("logprobs") or {}).get("content") or []
    for idx, item in enumerate(lp_items):
        if item.get("token") == "_" and idx >= 1 and idx + 3 < len(lp_items):
            # 1-token prefix before "_"
            tok1 = lp_items[idx - 1].get("token", "").strip()
            if tok1 in _PREFIX_SET:
                prefix_lp = lp_items[idx - 1].get("logprob", 0.0)
                digits_lp = sum(lp_items[j].get("logprob", 0.0) for j in range(idx, idx + 4))
                return {
                    "intent_confidence": round(math.exp(prefix_lp + digits_lp), 6),
                    "sub_agent_confidence": round(math.exp(prefix_lp), 6),
                }
            # 2-token prefix before "_" (e.g., PL+N, EQ+P, US+G)
            if idx >= 2:
                tok2 = lp_items[idx - 2].get("token", "").strip() + lp_items[idx - 1].get("token", "").strip()
                if tok2 in _PREFIX_SET:
                    prefix_lp = lp_items[idx - 2].get("logprob", 0.0) + lp_items[idx - 1].get("logprob", 0.0)
                    digits_lp = sum(lp_items[j].get("logprob", 0.0) for j in range(idx, idx + 4))
                    return {
                        "intent_confidence": round(math.exp(prefix_lp + digits_lp), 6),
                        "sub_agent_confidence": round(math.exp(prefix_lp), 6),
                    }
    return {"intent_confidence": None, "sub_agent_confidence": None}


def classify_utterance(utterance: str) -> dict:
    token = get_access_token()
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": utterance},
        ],
        "max_tokens": 64,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 5,
    }

    t0 = time.perf_counter()
    req = urllib.request.Request(
        VERTEX_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        raw_resp = json.loads(resp.read().decode("utf-8"))
    inference_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    choice = raw_resp["choices"][0]
    msg = choice.get("message") or {}
    # Combine content and reasoning fields in case vLLM's reasoning parser strips markdown code fences into reasoning
    content = ((msg.get("content") or "") + " " + (msg.get("reasoning") or "")).strip()
    usage = raw_resp.get("usage", {})
    conf = extract_confidence_metrics(choice)

    intent_id = None
    sub_agent = None

    # Parse intent_id via regex allow-list first (immune to markdown code fences)
    match = _INTENT_ID_REGEX.search(content)
    if match:
        candidate_id = f"{match.group(1)}_{match.group(2)}"
        if candidate_id in INTENT_MAP:
            intent_id = candidate_id
            # Deterministically resolve the canonical sub-agent from the 160-intent taxonomy
            sub_agent = INTENT_MAP[intent_id]["sub_agent"]

    # Fallback JSON parse if model returned only sub_agent
    if not sub_agent:
        for valid_sa in VALID_SUB_AGENTS:
            if valid_sa.lower() in content.lower():
                sub_agent = valid_sa
                break

    intent_meta = INTENT_MAP.get(intent_id, {})
    return {
        "intent_id": intent_id or "UNKNOWN",
        "intent_name": intent_meta.get("intent_name", "Unknown / Greeting"),
        "sub_agent": sub_agent or "NONE",
        "intent_confidence": conf["intent_confidence"],
        "sub_agent_confidence": conf["sub_agent_confidence"],
        "model": raw_resp.get("model", ROUTER_MODEL_NAME),
        "inference_latency_ms": inference_ms,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
    }


class ClassifierHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, body: dict):
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        # Mandatory security headers per secure web backend guidelines
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path in ("/", "/health", "/healthz"):
            self._send_json(
                200,
                {
                    "status": "ok",
                    "service": os.environ.get("K_SERVICE", "intent-router-service"),
                    "intents_loaded": len(INTENT_MAP),
                    "sub_agents": sorted(VALID_SUB_AGENTS),
                },
            )
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/classify":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 16384:
                self._send_json(400, {"error": "Invalid request payload size"})
                return

            raw_body = self.rfile.read(content_length).decode("utf-8")
            req_json = json.loads(raw_body)
            utterance = str(req_json.get("utterance", "")).strip()
            if not utterance or len(utterance) > 2000:
                self._send_json(400, {"error": "Field 'utterance' is required (1..2000 chars)"})
                return

            result = classify_utterance(utterance)
            self._send_json(200, result)
        except Exception as e:
            logger.error("Classification failed: %s", type(e).__name__)
            self._send_json(500, {"error": "Internal classification error"})


if __name__ == "__main__":
    # Listen on 127.0.0.1 for local testing; on Cloud Run bind to PORT
    host = os.environ.get("BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer((host, port), ClassifierHandler)
    logger.info("Starting Intent Router Service on %s:%d", host, port)
    server.serve_forever()
