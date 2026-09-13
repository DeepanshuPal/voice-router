"""Benchmark harness: run real audio through every live STT provider, record
latency + cost + (optional) word error rate, and score providers per language.

Usage:
    python -m benchmarks.harness --samples benchmarks/samples \
        [--references benchmarks/references] [--language en]

Samples are .wav files. References (optional) are same-named .txt transcripts;
without them, WER is skipped and scores rank on latency + cost only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from voice_router.config import load_config
from voice_router.providers.base import ProviderError, sine_wav
from voice_router.providers.registry import build_providers

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


def _edit_rate(ref: list, hyp: list) -> float:
    """Edit distance over sequences (words or characters), normalized by ref length."""
    if not ref:
        return 0.0
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)] / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Classic word error rate via edit distance on word sequences."""
    return _edit_rate(reference.lower().split(), hypothesis.lower().split())


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate: edit distance on characters, whitespace stripped.

    Word error rate is meaningless for scripts that are not segmented into
    whitespace-delimited words (Japanese, Chinese, Korean, Thai): a correct
    transcript with different phrase spacing scores WER 1.0. CER is the
    standard metric there.
    """
    ref = list("".join(reference.lower().split()))
    hyp = list("".join(hypothesis.lower().split()))
    return _edit_rate(ref, hyp)


# Languages scored with CER instead of WER (no whitespace word segmentation).
CHAR_ERROR_LANGS = {"ja", "zh", "ko", "th"}


async def run(samples: Path, references: Path | None, language: str, out: Path) -> dict:
    cfg = load_config()
    providers = build_providers(cfg)["stt"]
    wavs = sorted(samples.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"no .wav samples in {samples} - drop real clips there, "
                         "or run scripts/make_samples.py for synthetic ones")

    # Several providers here are async job APIs (upload, submit, poll), so a
    # strictly serial matrix takes hours. Run sample x provider pairs with a
    # small semaphore; latency is still measured per call, wall clock.
    sem = asyncio.Semaphore(6)

    async def one(wav: Path, name: str, adapter) -> dict:
        audio = wav.read_bytes()
        ref = None
        if references:
            ref_file = references / (wav.stem + ".txt")
            ref = ref_file.read_text().strip() if ref_file.exists() else None
        model = adapter.spec.models[0]
        async with sem:
            start = time.perf_counter()
            try:
                text = await adapter.transcribe(audio, model, language)
                latency = (time.perf_counter() - start) * 1000
                metric = "cer" if language in CHAR_ERROR_LANGS else "wer"
                err = cer(ref, text) if metric == "cer" else wer(ref, text)
                return {
                    "sample": wav.name, "provider": name, "model": model,
                    "language": language,
                    "audio_s": round(len(audio) / (16000 * 2), 2),
                    "latency_ms": round(latency, 1),
                    "cost_usd": adapter.spec.unit_cost(len(audio) / (16000 * 2 * 60)),
                    "wer": round(err, 4) if ref else None,
                    "metric": metric,
                    "status": "ok",
                }
            except ProviderError as e:
                return {"sample": wav.name, "provider": name, "model": model,
                        "language": language,
                        "status": "error", "detail": str(e)[:200]}

    tasks = [one(wav, name, adapter)
             for wav in wavs
             for name, adapter in providers.items()
             if adapter.supports(language)]
    runs = list(await asyncio.gather(*tasks))

    # Score per provider: 1 - WER when references exist, else inverse latency.
    scores: dict[str, dict[str, float]] = {"stt": {}}
    for name in providers:
        ok = [r for r in runs if r["provider"] == name and r["status"] == "ok"]
        if not ok:
            continue
        if any(r["wer"] is not None for r in ok):
            score = 1 - sum(r["wer"] or 0 for r in ok) / len(ok)
        else:
            score = 1 / (1 + sum(r["latency_ms"] for r in ok) / len(ok) / 1000)
        scores["stt"][name] = {language: round(score, 4)}

    payload = {"sample_data": False,
               "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "scores": scores, "runs": runs,
               "note": "regenerate with: python -m benchmarks.harness --samples benchmarks/samples"}
    out.write_text(json.dumps(payload, indent=2))
    return payload


def render_leaderboard(payload: dict) -> str:
    lines = ["| provider | language | score |", "|---|---|---|"]
    for name, langs in sorted(payload["scores"].get("stt", {}).items()):
        for lang, score in langs.items():
            lines.append(f"| {name} | {lang} | {score:.3f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, default=Path("benchmarks/samples"))
    ap.add_argument("--references", type=Path, default=None)
    ap.add_argument("--language", default="en")
    ap.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = ap.parse_args()

    payload = asyncio.run(run(args.samples, args.references, args.language, args.out))
    print(f"wrote {args.out}\n")
    print(render_leaderboard(payload))


if __name__ == "__main__":
    main()
