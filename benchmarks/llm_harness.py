"""LLM benchmark leg: time fixed prompts through the LLMs a voice agent would
actually use (fast, cheap, good enough), record latency, tokens/sec and cost.

This measures speed and price only - answer quality is not scored. Prompts are
fixed short customer-service turns, one per board language.

    python -m benchmarks.llm_harness --out /tmp/llm.json
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import json

# One fixed prompt per language: the same short support turn every model gets.
PROMPTS = {
    "en": "A customer calls saying their order arrived damaged. Reply in one short sentence.",
    "es": "Un cliente llama diciendo que su pedido llego danado. Responde en una frase corta.",
    "hi": "एक ग्राहक कॉल करके कहता है कि उसका ऑर्डर टूटा हुआ पहुंचा है। एक छोटे वाक्य में जवाब दें।",
    "fr": "Un client appelle parce que sa commande est arrivee abimee. Reponds en une courte phrase.",
    "de": "Ein Kunde ruft an, weil seine Bestellung beschaedigt ankam. Antworte in einem kurzen Satz.",
    "ja": "注文した商品が壊れて届いたとお客様から電話です。短い一文で答えてください。",
}

# (provider label, model id, env key, $/1M input tokens, $/1M output tokens)
MODELS = [
    ("groq", "openai/gpt-oss-120b", "GROQ_API_KEY", 0.15, 0.60),
    ("groq", "openai/gpt-oss-20b", "GROQ_API_KEY", 0.075, 0.30),
]

MAX_TOKENS = 120


async def one(client: httpx.AsyncClient, label: str, model: str, key: str,
              lang: str, prompt: str, pin: float, pout: float) -> dict:
    start = time.perf_counter()
    try:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "max_tokens": MAX_TOKENS,
                  "messages": [{"role": "user", "content": prompt}]},
        )
        latency = (time.perf_counter() - start) * 1000
        if resp.status_code != 200:
            return {"provider": label, "model": model, "language": lang,
                    "status": "error", "detail": f"{resp.status_code}: {resp.text[:160]}"}
        data = resp.json()
        usage = data.get("usage") or {}
        out_tokens = usage.get("completion_tokens") or 0
        in_tokens = usage.get("prompt_tokens") or 0
        return {
            "provider": label, "model": model, "language": lang,
            "latency_ms": round(latency, 1),
            "tokens_out": out_tokens,
            "tokens_per_s": round(out_tokens / (latency / 1000), 1) if latency else None,
            "cost_usd": round(in_tokens * pin / 1e6 + out_tokens * pout / 1e6, 8),
            "status": "ok",
        }
    except Exception as e:
        return {"provider": label, "model": model, "language": lang,
                "status": "error", "detail": str(e)[:160]}


async def run() -> dict:
    runs = []
    async with httpx.AsyncClient(timeout=60) as client:
        for label, model, env_key, pin, pout in MODELS:
            key = os.environ.get(env_key)
            if not key:
                continue
            sem = asyncio.Semaphore(2)

            async def gated(lg: str, pr: str) -> dict:
                async with sem:
                    return await one(client, label, model, key, lg, pr, pin, pout)

            runs += list(await asyncio.gather(*(gated(lg, pr) for lg, pr in PROMPTS.items())))
    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "runs": runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    payload = asyncio.run(run())
    ok = sum(1 for r in payload["runs"] if r["status"] == "ok")
    print(f"llm: {ok}/{len(payload['runs'])} ok")
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
