"""Arena audio generator: synthesize the shared reference texts through every
live TTS provider and save the clips the blind A/B arena plays.

Clips are real API output, written to docs/arena/audio/<provider>/<sample>.<ext>
with a manifest the arena page loads. English only for now - the one language
every wired provider shares.

    python -m benchmarks.arena_audio
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from voice_router.config import load_config
from voice_router.providers.base import ProviderError
from voice_router.providers.registry import build_providers

ROOT = Path(__file__).resolve().parent.parent
REFERENCES = ROOT / "benchmarks" / "references"
OUT = ROOT / "docs" / "arena" / "audio"
LANGUAGE = "en"  # en is the one language every wired TTS provider shares


def ext_for(data: bytes) -> str:
    if data[:4] == b"RIFF":
        return "wav"
    if data[:3] == b"ID3" or data[:2] == b"\xff\xfb":
        return "mp3"
    if data[:4] == b"OggS":
        return "ogg"
    return "bin"


async def main() -> None:
    cfg = load_config()
    providers = build_providers(cfg)["tts"]
    texts = []
    for ref in sorted(REFERENCES.glob(f"{LANGUAGE}-*.txt")):
        texts.append((ref.stem, ref.read_text().strip()))
    if not texts:
        raise SystemExit(f"no reference texts for {LANGUAGE}")

    manifest = {"generated_at": datetime.now(timezone.utc).isoformat(),
                "language": LANGUAGE, "clips": []}
    for name, adapter in providers.items():
        model = adapter.model_for(LANGUAGE)
        for stem, text in texts:
            try:
                audio = await adapter.synthesize(text, model, "alloy")
            except ProviderError as e:
                print(f"skip {name}/{stem}: {e}")
                continue
            ext = ext_for(audio)
            dest = OUT / name / f"{stem}.{ext}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(audio)
            manifest["clips"].append({"provider": name, "model": model,
                                      "sample": stem, "file": f"audio/{name}/{stem}.{ext}",
                                      "chars": len(text)})
            print(f"wrote {dest.relative_to(ROOT)} ({len(audio)} bytes)")
    (OUT.parent / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {len(manifest['clips'])} clips")


if __name__ == "__main__":
    asyncio.run(main())
