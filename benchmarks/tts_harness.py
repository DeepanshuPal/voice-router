"""TTS benchmark leg: synthesize fixed texts through every live TTS provider,
record latency + metered cost. Quality is subjective and is NOT scored here -
these are speed and price numbers. (Blind listening tests live in the arena.)

Input texts are FLEURS reference transcripts (benchmarks/references/) - the
same corpus the STT board measures, so both legs share one source of truth.

    python -m benchmarks.tts_harness --language en --out /tmp/tts.json
"""

from __future__ import annotations

import argparse
import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path

import json

from voice_router.config import load_config
from voice_router.providers.base import ProviderError
from voice_router.providers.registry import build_providers

ROOT = Path(__file__).resolve().parent.parent
REFERENCES = ROOT / "benchmarks" / "references"
SAMPLES_PER_LANG = 3  # <lang>-1..3.txt - small on purpose: free tiers are tight


async def run(language: str, samples_per_lang: int = SAMPLES_PER_LANG, repeats: int = 5) -> dict:
    cfg = load_config()
    providers = build_providers(cfg)["tts"]
    texts = []
    for i in range(1, samples_per_lang + 1):
        ref = REFERENCES / f"{language}-{i}.txt"
        if ref.exists():
            texts.append((ref.stem, ref.read_text().strip()))
    if not texts:
        raise SystemExit(f"no reference texts for {language} in {REFERENCES}")

    async def one(stem: str, text: str, name: str, adapter, repeat: int) -> dict:
        model = adapter.model_for(language)
        try:
            timing = await adapter.synthesize_timed(text, model, "alloy")
            return {
                "sample": stem, "provider": name, "model": model,
                "language": language, "chars": len(text), "repeat": repeat,
                "protocol": timing.protocol, "ttfb_ms": round(timing.ttfb_ms, 1) if timing.ttfb_ms is not None else None,
                "full_synthesis_wall_clock_ms": round(timing.completion_ms, 1),
                "audio_bytes": len(timing.audio),
                "cost_usd": adapter.spec.unit_cost(len(text) / 1000), "status": "ok",
            }
        except ProviderError as e:
            return {"sample": stem, "provider": name, "model": model,
                    "language": language, "chars": len(text), "repeat": repeat,
                    "status": "error", "detail": str(e)[:200]}

    # Serial by design. Streaming TTFB and batch completion are never blended.
    runs = []
    for stem, text in texts:
        for name, adapter in providers.items():
            if not adapter.supports(language):
                continue
            for repeat in range(1, repeats + 1):
                runs.append(await one(stem, text, name, adapter, repeat))

    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "runs": runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", default="en")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    payload = asyncio.run(run(args.language, repeats=args.repeats))
    ok = sum(1 for r in payload["runs"] if r["status"] == "ok")
    print(f"{args.language}: {ok}/{len(payload['runs'])} ok")
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
