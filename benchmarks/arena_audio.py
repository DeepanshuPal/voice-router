"""Incrementally fill the blind TTS arena from explicitly allowed providers.

Existing clips are preserved. The default is deliberately safe: no provider is
called unless it is named in ``--providers``. This keeps a gap-fill run from
silently spending quota on every configured key.

    python -m benchmarks.arena_audio --providers groq-tts,hume --missing-only
"""
from __future__ import annotations

import argparse
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
MANIFEST = OUT.parent / "manifest.json"
LANGUAGE = "en"


def ext_for(data: bytes) -> str:
    if data[:4] == b"RIFF": return "wav"
    if data[:3] == b"ID3" or (len(data)>1 and data[0]==0xFF and (data[1]&0xE0)==0xE0): return "mp3"
    if data[:4] == b"OggS": return "ogg"
    return "bin"


def load_manifest() -> dict:
    if MANIFEST.exists():
        data=json.loads(MANIFEST.read_text())
        data.setdefault("clips", [])
        return data
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "language": LANGUAGE, "clips": []}


async def run(provider_names: list[str], missing_only: bool=True) -> dict:
    if not provider_names:
        raise SystemExit("refusing to call every provider: pass --providers name[,name]")
    available=build_providers(load_config())["tts"]
    unknown=[name for name in provider_names if name not in available]
    if unknown: raise SystemExit(f"requested providers unavailable: {', '.join(unknown)}")
    texts=[(ref.stem,ref.read_text().strip()) for ref in sorted(REFERENCES.glob(f"{LANGUAGE}-*.txt"))]
    if not texts: raise SystemExit(f"no reference texts for {LANGUAGE}")
    manifest=load_manifest()
    existing={(c["provider"],c["sample"]) for c in manifest["clips"]}
    for name in provider_names:
        adapter=available[name]; model=adapter.model_for(LANGUAGE)
        for stem,text in texts:
            if missing_only and (name,stem) in existing:
                print(f"keep {name}/{stem}"); continue
            try:
                audio=await adapter.synthesize(text,model,"alloy")
            except ProviderError as exc:
                # One exhausted/disabled key should not burn 14 more calls.
                print(f"stop {name} after {stem}: {exc}")
                break
            ext=ext_for(audio); dest=OUT/name/f"{stem}.{ext}"
            dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(audio)
            manifest["clips"]=[c for c in manifest["clips"] if not (c["provider"]==name and c["sample"]==stem)]
            manifest["clips"].append({"provider":name,"model":model,"sample":stem,
                                      "file":f"audio/{name}/{stem}.{ext}","chars":len(text)})
            existing.add((name,stem)); print(f"wrote {dest.relative_to(ROOT)} ({len(audio)} bytes)")
    manifest["clips"].sort(key=lambda c:(c["provider"],c["sample"]))
    manifest["generated_at"]=datetime.now(timezone.utc).isoformat()
    MANIFEST.write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"manifest: {len(manifest['clips'])} clips")
    return manifest


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--providers",required=True,help="comma-separated provider allowlist")
    ap.add_argument("--missing-only",action=argparse.BooleanOptionalAction,default=True)
    args=ap.parse_args()
    asyncio.run(run([x.strip() for x in args.providers.split(",") if x.strip()],args.missing_only))

if __name__=="__main__": main()
