"""Run the STT benchmark matrix across all sample languages and merge results.

Discovers languages from benchmarks/samples/<lang>-<n>.wav prefixes, runs the
harness per language with matching references, and merges everything into one
benchmarks/results.json. Mock providers are disabled via a generated config so
CI runs (with real keys) never publish mock rows.

    python scripts/run_matrix.py
"""

from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import json

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "benchmarks" / "samples"
REFERENCES = ROOT / "benchmarks" / "references"
RESULTS = ROOT / "benchmarks" / "results.json"

import sys
sys.path.insert(0, str(ROOT))
from benchmarks.harness import run  # noqa: E402


def languages() -> list[str]:
    langs = set()
    for wav in SAMPLES.glob("*.wav"):
        m = re.match(r"([a-z]{2})-", wav.name)
        if m:
            langs.add(m.group(1))
    return sorted(langs)


def disable_mock() -> None:
    """Point ROUTER_CONFIG at a copy of providers.yaml with mocks off."""
    src = Path(os.environ.get("ROUTER_CONFIG", ROOT / "config" / "providers.yaml"))
    text = src.read_text().replace("enabled: true", "enabled: false")
    tmp = Path("/tmp/voice-router-no-mock.yaml")
    tmp.write_text(text)
    os.environ["ROUTER_CONFIG"] = str(tmp)


def main() -> None:
    disable_mock()
    runs, scores = [], {}
    for lang in languages():
        samp = Path(f"/tmp/matrix-samples-{lang}")
        ref = Path(f"/tmp/matrix-refs-{lang}")
        samp.mkdir(exist_ok=True)
        ref.mkdir(exist_ok=True)
        for f in samp.glob("*.wav"):
            f.unlink()
        for f in ref.glob("*.txt"):
            f.unlink()
        for wav in SAMPLES.glob(f"{lang}-*.wav"):
            (samp / wav.name).write_bytes(wav.read_bytes())
            txt = REFERENCES / (wav.stem + ".txt")
            if txt.exists():
                (ref / txt.name).write_text(txt.read_text())
        payload = asyncio.run(run(samp, ref if any(ref.glob("*.txt")) else None,
                                  lang, Path("/tmp/matrix-out.json")))
        runs += payload["runs"]
        for prov, langs_ in payload["scores"]["stt"].items():
            scores.setdefault(prov, {}).update(langs_)
        print(f"[matrix] {lang}: " +
              ", ".join(f"{p}={s[lang]:.3f}" for p, s in scores.items() if lang in s))

    merged = {"sample_data": False,
              "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "scores": {"stt": scores}, "runs": runs,
              "note": "regenerate with: python scripts/run_matrix.py"}
    RESULTS.write_text(json.dumps(merged, indent=2))
    ok = sum(1 for r in runs if r["status"] == "ok")
    print(f"wrote {RESULTS} ({ok}/{len(runs)} ok runs)")


if __name__ == "__main__":
    main()
