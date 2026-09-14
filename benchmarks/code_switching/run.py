"""Run the committed Hinglish corpus through every configured STT provider.

This is quota-consuming. Invoke it only from the scheduled/manual workflow.
"""
from __future__ import annotations
import argparse, asyncio, json, os, re, time, unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from voice_router.config import load_config
from voice_router.providers.base import ProviderError
from voice_router.providers.registry import build_providers

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = Path(__file__).with_name("samples")
REFERENCES = Path(__file__).with_name("references")
MANIFEST = Path(__file__).with_name("manifest.json")
RESULTS = Path(__file__).with_name("results.json")


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.findall(r"[a-z]+(?:['-][a-z]+)*|[\u0900-\u097f]+|\d+", text)


def edit_rate(ref: list[str], hyp: list[str]) -> float:
    if not ref: return 0.0
    d=list(range(len(hyp)+1))
    for i,r in enumerate(ref,1):
        prev,d[0]=d[0],i
        for j,h in enumerate(hyp,1):
            prev,d[j]=d[j],min(d[j]+1,d[j-1]+1,prev+(r!=h))
    return d[-1]/len(ref)


def recall(ref: list[str], hyp: list[str]) -> float | None:
    if not ref: return None
    rc,hc=Counter(ref),Counter(hyp)
    return sum(min(n,hc[t]) for t,n in rc.items())/sum(rc.values())


def metrics(reference: str, hypothesis: str) -> dict:
    ref,hyp=tokens(reference),tokens(hypothesis)
    is_en=lambda t: bool(re.fullmatch(r"[a-z]+(?:['-][a-z]+)*",t))
    is_hi=lambda t: any('\u0900'<=c<='\u097f' for c in t)
    en=recall([t for t in ref if is_en(t)],[t for t in hyp if is_en(t)])
    hi=recall([t for t in ref if is_hi(t)],[t for t in hyp if is_hi(t)])
    return {"mixed_wer":round(edit_rate(ref,hyp),4),
            "english_token_recall":round(en,4) if en is not None else None,
            "devanagari_token_recall":round(hi,4) if hi is not None else None,
            "script_balance":round((en+hi)/2,4) if en is not None and hi is not None else None}


async def run(out: Path) -> dict:
    cfg=load_config(); providers=build_providers(cfg)["stt"]
    manifest=json.loads(MANIFEST.read_text()); sem=asyncio.Semaphore(4)
    async def one(item,name,adapter):
        wav=SAMPLES/f"{item['id']}.wav"; ref=(REFERENCES/f"{item['id']}.txt").read_text().strip()
        started=time.perf_counter()
        try:
            async with sem: hyp=await adapter.transcribe(wav.read_bytes(),adapter.spec.models[0],"hi")
            return {"sample":item["id"],"provider":name,"model":adapter.spec.models[0],
                    "status":"ok","latency_ms":round((time.perf_counter()-started)*1000,1),
                    "audio_s":item["duration_s"],"reference":ref,"hypothesis":hyp,**metrics(ref,hyp)}
        except ProviderError as exc:
            return {"sample":item["id"],"provider":name,"model":adapter.spec.models[0],
                    "status":"error","detail":str(exc)[:200]}
    jobs=[one(item,name,a) for item in manifest["samples"] for name,a in providers.items() if a.supports("hi")]
    runs=list(await asyncio.gather(*jobs)); rows=[]
    for name in providers:
        ok=[r for r in runs if r["provider"]==name and r["status"]=="ok"]
        if not ok: continue
        avg=lambda k: round(sum(r[k] for r in ok if r[k] is not None)/sum(r[k] is not None for r in ok),4)
        rows.append({"provider":name,"model":ok[0]["model"],"samples":len(ok),
                     "mixed_wer":avg("mixed_wer"),"english_token_recall":avg("english_token_recall"),
                     "devanagari_token_recall":avg("devanagari_token_recall"),"script_balance":avg("script_balance"),
                     "p50_latency_ms":round(sorted(r["latency_ms"] for r in ok)[len(ok)//2],1)})
    rows.sort(key=lambda r:(-r["script_balance"],r["mixed_wer"]))
    payload={"generated_at":datetime.now(timezone.utc).isoformat(timespec="seconds"),"sample_data":False,
             "dataset":manifest,"methodology":{"language_hint":"hi","primary_metric":"script_balance","note":"Exact mixed-script references; English and Devanagari multiset token recall are averaged so transliteration cannot silently count as retention."},
             "results":rows,"runs":runs}
    out.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n"); return payload

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=RESULTS); a=ap.parse_args()
    p=asyncio.run(run(a.out)); print(f"wrote {a.out}: {len(p['results'])} providers, {len(p['runs'])} runs")
