"""Protocol-native streaming STT measurement. Never mixes with batch timing.

This harness is intentionally not run by CI or a schedule. It requires real
recorded audio, a pinned runner region and explicit manual invocation after the
measurement review gate opens.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.harness import CHAR_ERROR_LANGS, cer, corpus_error_rate, wer
from voice_router.config import load_config
from voice_router.providers.registry import build_providers


def _summary(runs: list[dict]) -> list[dict]:
    out=[]
    for provider in sorted({r["provider"] for r in runs}):
        rows=[r for r in runs if r["provider"]==provider and r["status"]=="ok"]
        if len(rows)<5: continue
        row={"provider":provider,"protocol":"streaming_websocket","n":len(rows)}
        for key in ("connection_open_ms","first_partial_ms","stable_partial_ms","final_ms","completion_ms","realtime_factor"):
            vals=[r[key] for r in rows if r.get(key) is not None]
            if len(vals)<5: row[key]=None; continue
            q1,_,q3=statistics.quantiles(vals,n=4,method="inclusive")
            row[key]={"median":round(statistics.median(vals),3),"q1":round(q1,3),"q3":round(q3,3),"iqr":round(q3-q1,3)}
        out.append(row)
    return out


async def run(samples: Path, references: Path, language: str, out: Path, repeats: int=5):
    if repeats < 5: raise SystemExit("streaming publication artifacts require at least five repeats")
    region=os.environ.get("BENCHMARK_REGION")
    if not region: raise SystemExit("BENCHMARK_REGION is required")
    providers={n:p for n,p in build_providers(load_config())["stt"].items()
               if n=="deepgram-flux"}
    if not providers: raise SystemExit("no streaming STT provider is configured with a key")
    runs=[]
    for wav in sorted(samples.glob("*.wav")):
        ref_path=references/(wav.stem+".txt")
        reference=ref_path.read_text().strip() if ref_path.exists() else None
        for name,provider in providers.items():
            if not provider.supports(language): continue
            model="flux-general-en" if language=="en" else "flux-general-multi"
            for repeat in range(1,repeats+1):
                try:
                    result=await provider.transcribe_streaming(wav.read_bytes(),model,language)
                    final_ms=result.first_ms("final")
                    metric="cer" if language in CHAR_ERROR_LANGS else "wer"
                    error=(cer(reference,result.transcript) if metric=="cer" else wer(reference,result.transcript,language)) if reference else None
                    runs.append({"sample":wav.name,"provider":name,"model":model,"language":language,"repeat":repeat,
                        "protocol":result.protocol,"connection_open_ms":result.first_ms("connection_open"),
                        "first_partial_ms":result.first_ms("first_partial"),"stable_partial_ms":result.first_ms("stable_partial"),
                        "final_ms":final_ms,"completion_ms":result.completion_ms,
                        "audio_duration_ms":result.audio_duration_ms,"realtime_factor":result.completion_ms/result.audio_duration_ms,
                        "reference":reference,"hypothesis":result.transcript,"metric":metric,"error":error,
                        "events":[{"kind":e.kind,"elapsed_ms":e.elapsed_ms,"transcript":e.transcript,"provider_data":e.provider_data} for e in result.events],"status":"ok"})
                except Exception as exc:
                    runs.append({"sample":wav.name,"provider":name,"model":model,"language":language,"repeat":repeat,"protocol":"streaming_websocket","status":"error","detail":str(exc)[:200]})
    accuracy={}
    for provider in providers:
        pairs=[(r["reference"],r["hypothesis"]) for r in runs if r["provider"]==provider and r["status"]=="ok" and r.get("reference")]
        if pairs: accuracy[provider]={"metric":"cer" if language in CHAR_ERROR_LANGS else "wer","corpus_error_rate":corpus_error_rate(pairs,language)}
    payload={"generated_at":datetime.now(timezone.utc).isoformat(timespec="seconds"),"measurement":{"runner_region":region,"execution":"serial","audio_delivery":"realtime-paced","chunk_ms":80,"repeats_per_clip":repeats,"publication":"withheld pending adversarial sign-off"},"accuracy":accuracy,"timing_summary":_summary(runs),"runs":runs}
    out.write_text(json.dumps(payload,indent=2)); return payload


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--samples',type=Path,required=True); ap.add_argument('--references',type=Path,required=True); ap.add_argument('--language',default='en'); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--repeats',type=int,default=5)
    args=ap.parse_args(); asyncio.run(run(args.samples,args.references,args.language,args.out,args.repeats))

if __name__=='__main__': main()
