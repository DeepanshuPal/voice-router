"""Generate the static benchmark leaderboard site from harness results.  # adapter-aware triggers

Reads benchmarks/results.json (produced by benchmarks/harness.py) and renders a
dependency-free static site into docs/ (served by GitHub Pages):

    python scripts/generate_leaderboard.py                 # results.json -> docs/
    python scripts/generate_leaderboard.py --results benchmarks/sample_results.json

If the results file has no real provider runs yet (empty, or only the built-in
mock providers), the generator falls back to benchmarks/sample_results.json so
the full loop renders end to end with zero provider keys. Sample data is always
rendered behind a visible SAMPLE DATA banner - it is never presented as a real
measurement.

Design: same visual language as the am-i-cited site (Geist Sans/Mono, hairline
borders, type labels, mono tabular numbers, #5e6ad2 accent), light mode.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "benchmarks" / "results.json"
DEFAULT_SAMPLE = ROOT / "benchmarks" / "sample_results.json"
DEFAULT_OUT = ROOT / "docs"

LANG_NAMES = {"en": "English", "es": "Spanish", "hi": "Hindi", "fr": "French",
              "de": "German", "pt": "Portuguese", "ja": "Japanese", "zh": "Chinese"}

GEIST_SANS = "https://cdn.jsdelivr.net/npm/geist@1.3.1/dist/fonts/geist-sans/Geist-Variable.woff2"
GEIST_MONO = "https://cdn.jsdelivr.net/npm/geist@1.3.1/dist/fonts/geist-mono/GeistMono-Variable.woff2"
REPO_URL = "https://github.com/DeepanshuPal/voice-router"


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text())


def has_real_runs(payload: dict) -> bool:
    """A payload counts as real only if it is not flagged as sample data and at
    least one ok run comes from a non-mock provider."""
    if payload.get("sample_data"):
        return False
    ok = [r for r in payload.get("runs", []) if r.get("status") == "ok"]
    return any(not r.get("provider", "").startswith("mock") for r in ok)


def aggregate(payload: dict) -> list[dict]:
    """One leaderboard row per (provider, model); languages as per-language
    submetrics rendered side by side."""
    rows: dict[tuple[str, str], dict] = {}
    scores = payload.get("scores", {}).get("stt", {})
    for run in payload.get("runs", []):
        if run.get("status") != "ok":
            continue
        key = (run["provider"], run.get("model", ""))
        row = rows.setdefault(key, {
            "provider": run["provider"], "model": key[1],
            "samples": 0, "wers": [], "latencies": [],
            "cost": 0.0, "audio_s": 0.0, "langs": {},
        })
        lang = run.get("language") or "en"
        lrow = row["langs"].setdefault(lang, {"samples": 0, "wers": [], "score": None})
        row["samples"] += 1
        lrow["samples"] += 1
        if run.get("wer") is not None:
            row["wers"].append(run["wer"])
            lrow["wers"].append(run["wer"])
        if run.get("latency_ms") is not None:
            row["latencies"].append(run["latency_ms"])
        row["cost"] += run.get("cost_usd") or 0.0
        row["audio_s"] += run.get("audio_s") or 0.0

    out = []
    for row in rows.values():
        row["avg_wer"] = (sum(row["wers"]) / len(row["wers"])) if row["wers"] else None
        row["p50_ms"] = statistics.median(row["latencies"]) if row["latencies"] else None
        row["cost_per_min"] = (row["cost"] / row["audio_s"] * 60) if row["audio_s"] else None
        lang_scores = []
        for lang, lrow in row["langs"].items():
            lrow["avg_wer"] = (sum(lrow["wers"]) / len(lrow["wers"])) if lrow["wers"] else None
            lrow["score"] = scores.get(row["provider"], {}).get(lang)
            if lrow["score"] is None and lrow["avg_wer"] is not None:
                lrow["score"] = round(1 - lrow["avg_wer"], 4)
            if lrow["score"] is not None:
                lang_scores.append(lrow["score"])
            del lrow["wers"]
        row["score"] = round(sum(lang_scores) / len(lang_scores), 4) if lang_scores else None
        out.append(row)
    out.sort(key=lambda r: -(r["score"] or 0))
    return out


def aggregate_tts(payload: dict) -> list[dict]:
    """One row per (provider, model): latency + cost, per-language breakdown."""
    rows: dict[tuple[str, str], dict] = {}
    for run in payload.get("tts_runs", []):
        if run.get("status") != "ok":
            continue
        key = (run["provider"], run.get("model", ""))
        row = rows.setdefault(key, {
            "provider": run["provider"], "model": key[1],
            "samples": 0, "chars": 0, "latencies": [], "cost": 0.0, "langs": {},
        })
        lang = run.get("language") or "en"
        lrow = row["langs"].setdefault(lang, {"samples": 0, "latencies": []})
        row["samples"] += 1
        lrow["samples"] += 1
        row["chars"] += run.get("chars") or 0
        if run.get("latency_ms") is not None:
            row["latencies"].append(run["latency_ms"])
            lrow["latencies"].append(run["latency_ms"])
        row["cost"] += run.get("cost_usd") or 0.0
    out = []
    for row in rows.values():
        row["p50_ms"] = statistics.median(row["latencies"]) if row["latencies"] else None
        row["cost_per_1k"] = (row["cost"] / row["chars"] * 1000) if row["chars"] else None
        for lrow in row["langs"].values():
            lrow["p50_ms"] = statistics.median(lrow["latencies"]) if lrow["latencies"] else None
            del lrow["latencies"]
        out.append(row)
    out.sort(key=lambda r: (r["p50_ms"] or 1e9))
    return out


def aggregate_llm(payload: dict) -> list[dict]:
    """One row per (provider, model): p50 latency, tokens/sec, cost."""
    rows: dict[tuple[str, str], dict] = {}
    for run in payload.get("llm_runs", []):
        if run.get("status") != "ok":
            continue
        key = (run["provider"], run.get("model", ""))
        row = rows.setdefault(key, {
            "provider": run["provider"], "model": key[1],
            "samples": 0, "latencies": [], "tps": [], "cost": 0.0,
        })
        row["samples"] += 1
        if run.get("latency_ms") is not None:
            row["latencies"].append(run["latency_ms"])
        if run.get("tokens_per_s") is not None:
            row["tps"].append(run["tokens_per_s"])
        row["cost"] += run.get("cost_usd") or 0.0
    out = []
    for row in rows.values():
        row["p50_ms"] = statistics.median(row["latencies"]) if row["latencies"] else None
        row["avg_tps"] = (sum(row["tps"]) / len(row["tps"])) if row["tps"] else None
        row["cost_per_call"] = (row["cost"] / row["samples"]) if row["samples"] else None
        out.append(row)
    out.sort(key=lambda r: (r["p50_ms"] or 1e9))
    return out


# Providers we benchmark when keys exist. Ones missing from a run are listed
# with the reason, so absence is stated rather than silent.
# Full provider sweep target: every name on Speko's public benchmark
# (benchmarks.speko.ai, checked 2026-09-14) plus the major clouds. Each is
# either in the run with real numbers or listed here with the honest reason.
KNOWN_STT = {
    "deepgram": "",
    "assemblyai": "",
    "gladia": "",
    "speechmatics": "",
    "rev": "",
    "groq-whisper": "",
    "cartesia-stt": "",
    "openai-whisper": "no free tier - GPT-4o/GPT-4o-mini Transcribe, GPT Live Transcribe and Whisper-1 all need a paid OpenAI account",
    "elevenlabs-scribe": "ElevenLabs disabled free-tier API access on the account (anti-abuse flag, datacenter IP) - Scribe v2 needs a paid plan",
    "smallest-pulse": "Speko-listed (Pulse) - signup queued, free tier not verified yet",
    "xai-grok-stt": "Speko-listed (Grok STT) - signup not attempted yet",
    "inworld-stt": "Speko-listed (Realtime STT-1) - signup not attempted yet",
    "alibaba-qwen3-asr": "Speko-listed (Qwen3-ASR) - signup not attempted yet",
    "modulate-velma": "Speko-listed (Velma 2) - signup not attempted yet",
    "soniox": "Speko-listed (stt-rt-v5) - signup not attempted yet",
    "gradium": "Speko-listed (Gradium ASR) - signup not attempted yet",
    "google-chirp": "Chirp 3 needs a billed Google Cloud account (card required) - excluded by the $0/no-card rule",
    "gemini-transcribe": "Speko-listed (Gemini 3.5 Transcribe Live) - signup not attempted yet",
    "azure-speech": "Azure signup requires a credit card - excluded by the $0/no-card rule",
    "aws-transcribe": "AWS signup requires a credit card - excluded by the $0/no-card rule",
    "deepgram-flux": "Flux is streaming-only - this harness measures batch transcription, so a number would not be comparable",
}
KNOWN_TTS = {
    "elevenlabs": "API key works, but ElevenLabs disabled free-tier access on the account (anti-abuse flag, datacenter IP) - needs a paid plan",
    "openai-tts": "no free tier - needs a paid OpenAI account",
    "cartesia": "",
    "deepgram-aura": "",
    "groq-tts": "",
    "rime": "",
    "sarvam": "signup works (no card), but no free credits landed on the account (balance Rs 0, API returns 402) - needs paid credits",
    "hume": "signup not attempted yet",
    "smallest": "signup not attempted yet",
    "play.ht": "signup not attempted yet",
    "azure-speech": "Azure signup requires a credit card - excluded by the $0/no-card rule",
    "google-cloud-tts": "needs a billed Google Cloud account (card required) - excluded by the $0/no-card rule",
    "aws-polly": "AWS signup requires a credit card - excluded by the $0/no-card rule",
}
KNOWN_LLM = {
    "openrouter": "free models exist - key not wired into the run yet",
}


# ---------------------------------------------------------------- HTML helpers

CSS = """
@font-face{font-family:'Geist Sans';src:url('%SANS%') format('woff2');font-weight:100 900;font-style:normal;font-display:swap}
@font-face{font-family:'Geist Mono';src:url('%MONO%') format('woff2');font-weight:100 900;font-style:normal;font-display:swap}
:root{color-scheme:light;--hairline:rgba(0,0,0,.08);--border:#e7e7e7;--ink:#0a0a0a;--ink2:#404040;--mut:#8a8a8a;--faint:#a3a3a3;--accent:#5e6ad2;--accent-bright:#7c86e8;--good:#0e9f6e;--bad:#e5484d;--warn:#b45309}
*{box-sizing:border-box;margin:0;padding:0}
html{background:#fff;scroll-behavior:smooth}
body{background:#fff;color:var(--ink);font-family:'Geist Sans',system-ui,-apple-system,sans-serif;line-height:1.55;-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;min-height:100vh;display:flex;flex-direction:column}
::selection{background:rgba(94,106,210,.25)}
a{color:inherit;text-decoration:none}
code,.mono{font-family:'Geist Mono',ui-monospace,monospace}
.num{font-family:'Geist Mono',ui-monospace,monospace;font-variant-numeric:tabular-nums}
.wrap{width:100%;max-width:1152px;margin:0 auto;padding:0 24px}
main{flex:1}
.type-label{font-family:'Geist Mono',ui-monospace,monospace;font-size:11px;text-transform:uppercase;letter-spacing:.14em;color:var(--mut)}
.link-quiet{color:var(--ink2);transition:color .15s}
.link-quiet:hover{color:var(--ink)}

/* nav */
.nav{position:sticky;top:0;z-index:50;border-bottom:1px solid var(--hairline);background:rgba(255,255,255,.8);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px)}
.nav-inner{display:flex;align-items:center;justify-content:space-between;height:56px}
.brand{display:flex;align-items:center;gap:10px;font-size:15px;font-weight:500;letter-spacing:-.01em;white-space:nowrap}
.brand-mark{display:flex;align-items:center;justify-content:center;width:20px;height:20px;border:1px solid #d4d4d4;border-radius:5px;background:#f5f5f5;flex:none}
.brand-mark span{width:6px;height:6px;border-radius:9999px;background:var(--accent-bright)}
.nav-links{display:flex;align-items:center;gap:2px;font-size:13px}
.nav-links a{padding:6px 12px;border-radius:6px;color:var(--ink2);transition:color .15s,background .15s;white-space:nowrap}
.nav-links a:hover{color:var(--ink);background:#f5f5f5}
.nav-links a.active{color:var(--ink);font-weight:500}
.gh-btn{display:flex;align-items:center;gap:6px;margin-left:8px;padding:6px 12px;border:1px solid var(--border);border-radius:6px;background:#fff;color:var(--ink);font-size:13px;transition:background .15s,border-color .15s}
.gh-btn:hover{background:#fafafa;border-color:#d4d4d4}
.gh-btn svg{width:14px;height:14px}

/* hero */
.hero{display:flex;flex-wrap:wrap;align-items:flex-end;justify-content:space-between;gap:24px;padding:56px 0 0}
.hero h1{margin-top:12px;font-size:30px;font-weight:500;letter-spacing:-.02em;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.chip{display:inline-block;border:1px solid var(--border);background:#fafafa;border-radius:6px;padding:4px 8px;font-family:'Geist Mono',ui-monospace,monospace;font-size:11px;font-weight:400;color:var(--mut);letter-spacing:0}
.hero-sub{margin-top:8px;font-size:14px;color:var(--mut);max-width:560px}
.hero-meta{text-align:right;font-family:'Geist Mono',ui-monospace,monospace;font-size:11.5px;line-height:1.7;color:var(--mut)}

/* metric cards */
.metrics{margin-top:40px;display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
.metric{border:1px solid var(--border);border-radius:12px;background:#fff;padding:20px}
.metric .v{margin-top:12px;font-family:'Geist Mono',ui-monospace,monospace;font-variant-numeric:tabular-nums;font-size:32px;line-height:1;letter-spacing:-.02em}
.metric .s{margin-top:8px;font-size:12.5px;color:var(--mut)}

/* sections */
section{margin-top:56px}
.sec-head{margin-bottom:20px}
.sec-head h2{margin-top:8px;font-size:20px;font-weight:500;letter-spacing:-.01em}
.sec-head p{margin-top:8px;max-width:640px;font-size:13px;line-height:1.6;color:var(--mut)}

/* cards + tables */
.card{border:1px solid var(--border);border-radius:12px;background:#fff;overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{padding:12px 16px;text-align:right;border-bottom:1px solid var(--hairline);white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
th{font-family:'Geist Mono',ui-monospace,monospace;font-size:11px;text-transform:uppercase;letter-spacing:.14em;color:var(--mut);font-weight:400}
td.num,td{color:var(--ink2)}
td:first-child{color:var(--ink);font-weight:500}
tbody tr{transition:background .12s}
tbody tr:hover{background:#fafafa}
tbody tr:last-child td{border-bottom:none}
td .best,span.best{color:var(--good);font-weight:600}
.cellsub{font-family:'Geist Mono',ui-monospace,monospace;font-size:10.5px;color:var(--faint);margin-top:2px}
table.defs td{white-space:normal;vertical-align:top;line-height:1.6}
table.defs td:first-child{font-family:'Geist Mono',ui-monospace,monospace;font-size:12.5px;font-weight:400;color:var(--ink)}

/* sample banner */
.banner{margin-top:32px;border:1px solid #fcd34d;background:#fffbeb;color:#92400e;border-radius:10px;padding:12px 16px;font-size:13px;font-weight:500}
.banner a{text-decoration:underline;text-underline-offset:3px}

/* bar charts */
.charts{display:grid;grid-template-columns:1fr;gap:16px}
.chart-card{border:1px solid var(--border);border-radius:12px;background:#fff;padding:20px}
.chart-card h3{font-size:14px;font-weight:500;letter-spacing:-.01em;margin-bottom:4px}
.chart-card .hint{font-size:12px;color:var(--faint);margin-bottom:16px}
.bar-row{display:flex;align-items:center;gap:12px;margin:10px 0;font-size:13px}
.bar-label{width:190px;color:var(--ink2);flex:none;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar-track{flex:1;background:#f0f0f0;border-radius:9999px;height:8px;overflow:hidden}
.bar-fill{height:100%;border-radius:9999px}
.bar-val{width:96px;text-align:right;color:var(--mut);flex:none;font-family:'Geist Mono',ui-monospace,monospace;font-variant-numeric:tabular-nums;font-size:12.5px}

/* honesty / note card */
.note{margin-top:56px;border:1px solid var(--border);border-radius:12px;background:#fafafa;padding:24px}
.note p{margin-top:12px;max-width:760px;font-size:13.5px;line-height:1.65;color:var(--mut)}
.note a{color:var(--ink2);text-decoration:underline;text-decoration-color:#d4d4d4;text-underline-offset:3px}

/* methodology body copy */
.prose p{margin-bottom:10px;font-size:13.5px;line-height:1.65;color:var(--ink2);max-width:760px}
.prose code{font-size:12.5px;background:#f5f5f5;border:1px solid var(--hairline);border-radius:4px;padding:1px 5px}
.prose pre{background:#fafafa;border:1px solid var(--border);border-radius:8px;padding:14px;font-family:'Geist Mono',ui-monospace,monospace;font-size:12.5px;overflow-x:auto;margin:10px 0;color:var(--ink2)}
.card-pad{padding:24px}

/* footer */
footer{margin-top:80px;border-top:1px solid var(--hairline)}
.foot-inner{display:flex;flex-wrap:wrap;gap:24px;align-items:flex-end;justify-content:space-between;padding:40px 0}
.foot-name{font-size:13px;color:var(--ink)}
.foot-desc{margin-top:4px;max-width:380px;font-size:13px;line-height:1.6;color:var(--mut)}
.foot-desc a{text-decoration:underline;text-decoration-color:#d4d4d4;text-underline-offset:3px;color:var(--ink2)}
.foot-links{display:flex;align-items:center;gap:20px;font-family:'Geist Mono',ui-monospace,monospace;font-size:12px;color:var(--mut)}
.foot-links a{display:flex;align-items:center;gap:6px;color:var(--mut);transition:color .15s}
.foot-links a:hover{color:var(--ink)}
.foot-links svg{width:12px;height:12px}
.foot-meta{padding:0 0 24px;font-family:'Geist Mono',ui-monospace,monospace;font-size:11px;color:var(--faint)}
.foot-meta code{font-size:11px}

@media(max-width:900px){.metrics{grid-template-columns:repeat(2,1fr)}}
@media(max-width:640px){
  .optcol{display:none}
  .hero{padding-top:40px}
  .hero h1{font-size:24px}
  .hero-meta{text-align:left}
  .bar-label{width:120px;font-size:12px}
  .bar-val{width:84px;font-size:11.5px}
  .metric .v{font-size:26px}
  th,td{padding:10px 12px}
}
"""

GITHUB_MARK = ('<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">'
               '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 '
               "0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13"
               "-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66"
               ".07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15"
               "-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 "
               "2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 "
               '2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 '
               '2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/></svg>')


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def css() -> str:
    return CSS.replace("%SANS%", GEIST_SANS).replace("%MONO%", GEIST_MONO)


def nav(active: str) -> str:
    def link(href: str, label: str, key: str) -> str:
        cls = ' class="active"' if active == key else ""
        return f'<a href="{href}"{cls}>{label}</a>'
    return f"""<div class="nav"><div class="wrap nav-inner">
  <a class="brand" href="index.html"><span class="brand-mark"><span></span></span>voice-router</a>
  <div class="nav-links">
    {link("index.html", "Leaderboard", "index")}
    {link("methodology.html", "Methodology", "methodology")}
    {link("arena/", "Arena", "arena")}
    <a class="gh-btn" href="{REPO_URL}" target="_blank" rel="noreferrer">{GITHUB_MARK}<span>Star</span></a>
  </div>
</div></div>"""


def footer(generated_at: str, source: str | None) -> str:
    meta = f"generated {esc(generated_at)}"
    if source:
        meta += (f" from <code>{esc(source)}</code> &middot; refresh: "
                 f"<code>python scripts/generate_leaderboard.py</code>")
    return f"""<footer><div class="wrap">
  <div class="foot-inner">
    <div>
      <p class="foot-name">voice-router</p>
      <p class="foot-desc">Open-source router for voice AI. Benchmarks STT providers per language on accuracy, latency and cost, then routes calls to the best one. Built on <a href="https://github.com/BerriAI/litellm" target="_blank" rel="noreferrer">LiteLLM</a> and <a href="https://github.com/pipecat-ai/pipecat" target="_blank" rel="noreferrer">Pipecat</a>.</p>
    </div>
    <div class="foot-links">
      <a href="{REPO_URL}" target="_blank" rel="noreferrer">{GITHUB_MARK} source</a>
      <a href="data/leaderboard.json">json</a>
      <a href="llms.txt">llms.txt</a>
      <span>MIT</span>
    </div>
  </div>
  <div class="foot-meta">{meta}</div>
</div></footer>"""


def page(title: str, active: str, body: str, generated_at: str, source: str | None = None) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} - voice-router benchmarks</title>
<meta name="description" content="STT providers ranked by accuracy, latency, and cost per language - measured on real audio, not marketed.">
<style>{css()}</style>
</head>
<body>
{nav(active)}
<main><div class="wrap">
{body}
</div></main>
{footer(generated_at, source)}
</body>
</html>
"""


def fmt_wer(w) -> str:
    return "&mdash;" if w is None else f"{w * 100:.1f}%"


def fmt_cost(c) -> str:
    return "&mdash;" if c is None else f"${c:.4f}"


def fmt_ms(ms) -> str:
    return "&mdash;" if ms is None else f"{ms:.0f} ms"


def not_run_html(rows: list[dict]) -> str:
    ran = {r["provider"] for r in rows}
    items = [f"<li><code>{esc(name)}</code> - {esc(why)}</li>"
             for name, why in KNOWN_STT.items() if name not in ran and why]
    if not items:
        return ""
    return ('<p style="margin-top:12px;font-size:12.5px;color:var(--mut)">Not in this run:</p>'
            f'<ul style="margin-top:4px;font-size:12.5px;color:var(--mut);padding-left:20px">{"".join(items)}</ul>')


def render_table(rows: list[dict]) -> str:
    langs = sorted({lang for r in rows for lang in r["langs"]})
    best_score = max((r["score"] for r in rows if r["score"] is not None), default=None)
    best_by_lang = {}
    for lang in langs:
        vals = [r["langs"][lang]["score"] for r in rows
                if lang in r["langs"] and r["langs"][lang]["score"] is not None]
        if vals:
            best_by_lang[lang] = max(vals)

    body_rows = []
    for r in rows:
        score = r["score"]
        score_cell = "&mdash;" if score is None else f"{score:.3f}"
        if score is not None and score == best_score:
            score_cell = f'<span class="best">{score_cell}</span>'
        lang_cells = []
        for lang in langs:
            lrow = r["langs"].get(lang)
            if not lrow or lrow["avg_wer"] is None:
                lang_cells.append("<td>&mdash;</td>")
            else:
                cell = fmt_wer(lrow["avg_wer"])
                if lrow["score"] is not None and lrow["score"] == best_by_lang.get(lang):
                    cell = f'<span class="best">{cell}</span>'
                lang_cells.append(f'<td>{cell}<div class="cellsub">n={lrow["samples"]}</div></td>')
        body_rows.append(
            "<tr>"
            f"<td>{esc(r['provider'])}</td>"
            f'<td class="mono" style="font-size:12.5px;color:var(--mut)">{esc(r["model"])}</td>'
            f'<td class="num">{score_cell}</td>'
            f'<td class="num">{fmt_wer(r["avg_wer"])}</td>'
            f'<td class="num">{fmt_ms(r["p50_ms"])}</td>'
            f'<td class="num optcol">{fmt_cost(r["cost_per_min"])}</td>'
            f'<td class="num optcol">{r["samples"]}</td>'
            + "".join(lang_cells) +
            "</tr>"
        )
    cer_langs = {"ja", "zh", "ko", "th"}
    lang_headers = "".join(
        f"<th>{esc(lang)} {'CER' if lang in cer_langs else 'WER'}</th>" for lang in langs)
    return (
        "<table><thead><tr>"
        "<th>Provider</th><th>Model</th><th>Score</th>"
        "<th>WER</th><th>p50 latency</th>"
        '<th class="optcol">Cost/min</th><th class="optcol">Runs</th>'
        + lang_headers +
        "</tr></thead><tbody>" + "".join(body_rows) + "</tbody></table>"
        + not_run_html(rows)
    )


def bar_chart(title: str, hint: str, items: list[tuple[str, float, str]]) -> str:
    """items: (label, value, formatted value), pre-sorted best-first. The best
    bar gets the accent color; the rest stay neutral."""
    if not items:
        return ""
    peak = max(v for _, v, _ in items) or 1.0
    rows = []
    for i, (label, value, shown) in enumerate(items):
        pct = max(2.0, value / peak * 100)
        color = "var(--accent)" if i == 0 else "#d4d4d4"
        rows.append(
            f'<div class="bar-row"><div class="bar-label">{esc(label)}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div></div>'
            f'<div class="bar-val">{esc(shown)}</div></div>'
        )
    return (f'<div class="chart-card"><h3>{esc(title)}</h3>'
            f'<div class="hint">{esc(hint)}</div>{"".join(rows)}</div>')



def render_tts_section(rows: list[dict]) -> str:
    langs = sorted({lang for r in rows for lang in r["langs"]})
    body_rows = []
    for r in rows:
        lang_cells = []
        for lang in langs:
            lrow = r["langs"].get(lang)
            if not lrow or lrow.get("p50_ms") is None:
                lang_cells.append("<td>&mdash;</td>")
            else:
                lang_cells.append(f'<td class="num">{fmt_ms(lrow["p50_ms"])}<div class="cellsub">n={lrow["samples"]}</div></td>')
        body_rows.append(
            "<tr>"
            f"<td>{esc(r['provider'])}</td>"
            f'<td class="mono" style="font-size:12.5px;color:var(--mut)">{esc(r["model"])}</td>'
            f'<td class="num">{fmt_ms(r["p50_ms"])}</td>'
            f'<td class="num">{("$" + format(r["cost_per_1k"], ".3f")) if r["cost_per_1k"] is not None else "&mdash;"}</td>'
            f'<td class="num optcol">{r["samples"]}</td>'
            + "".join(lang_cells) +
            "</tr>"
        )
    ran = {r["provider"] for r in rows}
    not_run = [f"<li><code>{esc(name)}</code> - {esc(why)}</li>"
               for name, why in KNOWN_TTS.items() if name not in ran and why]
    table = ""
    if rows:
        lang_headers = "".join(f"<th>{esc(lang)} p50</th>" for lang in langs)
        table = ("<div class=\"card\"><table><thead><tr>"
                 "<th>Provider</th><th>Model</th><th>p50 latency</th>"
                 "<th>Cost/1k chars</th>"
                 '<th class="optcol">Runs</th>' + lang_headers +
                 "</tr></thead><tbody>" + "".join(body_rows) + "</tbody></table></div>")
    not_run_html = (f'<p style="margin-top:12px;font-size:12.5px;color:var(--mut)">Not in this run:</p>'
                    f'<ul style="margin-top:4px;font-size:12.5px;color:var(--mut);padding-left:20px">{"".join(not_run)}</ul>'
                    if not_run else "")
    empty = '<p style="font-size:13px;color:var(--mut)">No TTS provider keys in this run.</p>' if not rows else ""
    return f"""
<section>
  <div class="sec-head">
    <p class="type-label">text to speech</p>
    <h2>TTS: speed and price, no quality score</h2>
    <p>TTS quality is subjective - this board measures latency and metered cost only, and says so. Blind listening tests (an arena) are how quality gets measured honestly, and they are next. Synthesis input: 3 FLEURS reference transcripts per language, so both legs measure the same corpus. Cost is the provider's published per-character price over the characters actually synthesized.</p>
  </div>
  {table}{empty}{not_run_html}
</section>
"""


def render_llm_section(rows: list[dict]) -> str:
    body_rows = []
    for r in rows:
        body_rows.append(
            "<tr>"
            f"<td>{esc(r['provider'])}</td>"
            f'<td class="mono" style="font-size:12.5px;color:var(--mut)">{esc(r["model"])}</td>'
            f'<td class="num">{fmt_ms(r["p50_ms"])}</td>'
            f'<td class="num">{format(r["avg_tps"], ".0f") if r["avg_tps"] is not None else "&mdash;"}</td>'
            f'<td class="num">{("$" + format(r["cost_per_call"] * 1000, ".4f")) if r["cost_per_call"] is not None else "&mdash;"}</td>'
            f'<td class="num optcol">{r["samples"]}</td>'
            "</tr>"
        )
    ran = {r["provider"] for r in rows}
    not_run = [f"<li><code>{esc(name)}</code> - {esc(why)}</li>"
               for name, why in KNOWN_LLM.items() if name not in ran and why]
    table = ""
    if rows:
        table = ("<div class=\"card\"><table><thead><tr>"
                 "<th>Provider</th><th>Model</th><th>p50 latency</th>"
                 "<th>tokens/sec</th><th>Cost/1k calls</th>"
                 '<th class="optcol">Runs</th>'
                 "</tr></thead><tbody>" + "".join(body_rows) + "</tbody></table></div>")
    not_run_html = (f'<p style="margin-top:12px;font-size:12.5px;color:var(--mut)">Not in this run:</p>'
                    f'<ul style="margin-top:4px;font-size:12.5px;color:var(--mut);padding-left:20px">{"".join(not_run)}</ul>'
                    if not_run else "")
    empty = '<p style="font-size:13px;color:var(--mut)">No LLM keys in this run.</p>' if not rows else ""
    return f"""
<section>
  <div class="sec-head">
    <p class="type-label">llm leg</p>
    <h2>The LLM a voice turn waits on</h2>
    <p>Fixed short customer-service prompts, one per board language, 120 max tokens out. Measured: p50 latency to a complete reply, output tokens/sec, metered cost. Not measured: answer quality - one-line support replies are a speed test, not an intelligence test.</p>
  </div>
  {table}{empty}{not_run_html}
</section>
"""


def render_index(rows: list[dict], sample: bool, generated_at: str, source: str, run_url: str | None = None, tts_rows: list[dict] | None = None, llm_rows: list[dict] | None = None) -> str:
    banner = ""
    if sample:
        banner = ('<div class="banner">SAMPLE DATA - these numbers are synthetic and exist '
                  "to demonstrate the format. They are not real measurements. "
                  'See <a href="methodology.html">methodology</a>.</div>')

    total_samples = sum(r["samples"] for r in rows)
    n_providers = len(rows)
    n_langs = len({lang for r in rows for lang in r["langs"]})
    run_link = (f' &middot; <a class="link-quiet" style="text-decoration:underline;text-decoration-color:#d4d4d4;text-underline-offset:3px" '
                f'href="{esc(run_url)}" target="_blank" rel="noreferrer">action run</a>' if run_url else "")

    def best(key, reverse=False):
        vals = [r for r in rows if r[key] is not None]
        return (max if reverse else min)(vals, key=lambda r: r[key]) if vals else None

    top = best("score", reverse=True)
    bw = best("avg_wer")
    bl = best("p50_ms")
    bc = best("cost_per_min")

    def metric(label, value, sub):
        return (f'<div class="metric"><p class="type-label">{label}</p>'
                f'<p class="v">{value}</p><p class="s">{sub}</p></div>')

    metrics = '<div class="metrics">' + "".join([
        metric("top score", f"{top['score']:.3f}" if top else "-",
               f"{esc(top['provider'])} &middot; {esc(top['model'])}" if top else "no runs yet"),
        metric("best wer", fmt_wer(bw["avg_wer"]) if bw else "-",
               f"{esc(bw['provider'])} &middot; {esc(bw['model'])}" if bw else ""),
        metric("fastest p50", fmt_ms(bl["p50_ms"]) if bl else "-",
               f"{esc(bl['provider'])} &middot; {esc(bl['model'])}" if bl else ""),
        metric("lowest cost", (f"${bc['cost_per_min']:.4f}" if bc else "-"),
               f"per min &middot; {esc(bc['provider'])}" if bc else ""),
    ]) + "</div>"

    wer_items = sorted(
        [(f"{r['provider']} ({r['model']})", r["avg_wer"], fmt_wer(r["avg_wer"]).replace("&mdash;", "n/a"))
         for r in rows if r["avg_wer"] is not None],
        key=lambda x: x[1])
    lat_items = sorted(
        [(f"{r['provider']} ({r['model']})", r["p50_ms"], f"{r['p50_ms']:.0f} ms")
         for r in rows if r["p50_ms"] is not None],
        key=lambda x: x[1])
    score_items = sorted(
        [(f"{r['provider']} ({r['model']})", r["score"], f"{r['score']:.3f}")
         for r in rows if r["score"] is not None],
        key=lambda x: -x[1])

    body = f"""
<div class="hero">
  <div>
    <p class="type-label">benchmarks</p>
    <h1>Voice AI leaderboard <span class="chip">measured, not marketed</span></h1>
    <p class="hero-sub">Speech-to-text, text-to-speech and LLM providers ranked on measured runs - every number out of a harness, on real FLEURS audio and fixed prompts. Full rules on the <a class="link-quiet" style="text-decoration:underline;text-decoration-color:#d4d4d4;text-underline-offset:3px" href="methodology.html">methodology page</a>.</p>
  </div>
  <div class="hero-meta">
    <p>last run {esc(generated_at[:10])}{run_link}</p>
    <p>{total_samples} runs &middot; {n_providers} providers &middot; {n_langs} languages</p>
    <p>dataset: <a class="link-quiet" style="text-decoration:underline;text-decoration-color:#d4d4d4;text-underline-offset:3px" href="https://huggingface.co/datasets/google/fleurs" target="_blank" rel="noreferrer">Google FLEURS</a> &middot; 15 clips/language</p>
  </div>
</div>
{banner}
{metrics}
<section>
  <div class="sec-head">
    <p class="type-label">leaderboard</p>
    <h2>One row per provider, WER per language (CER for ja)</h2>
    <p>Score blends per-language accuracy into the number the router's benchmark strategy routes on. Green marks the best in each column. Every language cell averages 15 real FLEURS clips (n=15 under each number); the runs column is the provider total. Receipts on the <a class="link-quiet" style="text-decoration:underline;text-decoration-color:#d4d4d4;text-underline-offset:3px" href="methodology.html">methodology page</a>.</p>
  </div>
  <div class="card">{render_table(rows)}</div>
</section>
<section>
  <div class="sec-head">
    <p class="type-label">head to head</p>
    <h2>Same audio, every provider</h2>
  </div>
  <div class="charts">
    {bar_chart("Overall score", "all languages, higher is better", score_items)}
    {bar_chart("Word error rate", "all languages, lower is better", wer_items)}
    {bar_chart("p50 transcription latency", "all languages, lower is better", lat_items)}
  </div>
</section>
{render_tts_section(tts_rows or [])}
{render_llm_section(llm_rows or [])}
<div class="note">
  <p class="type-label">read these numbers like this</p>
  <p>Every run pushes the same FLEURS clips through each provider and records WER, latency and metered cost per sample - nothing is hand-edited or self-reported. A provider that does not support a language simply has no column entry. Clips, references, harness and raw <code>results.json</code> are all in the <a href="https://github.com/DeepanshuPal/voice-router" target="_blank" rel="noreferrer">repo</a>. Re-run it yourself: clone, add keys, run the harness.</p>
</div>
"""
    return page("Leaderboard", "index", body, generated_at, source)


def render_methodology(sample: bool, generated_at: str, run_url: str | None = None) -> str:
    note = (
        '<div class="banner">The current leaderboard shows SAMPLE DATA - synthetic numbers '
        "demonstrating the format while real provider runs are being collected.</div>"
        if sample else ""
    )
    body = f"""
<div class="hero">
  <div>
    <p class="type-label">methodology</p>
    <h1>How the numbers are made</h1>
    <p class="hero-sub">Every figure on the leaderboard comes out of a harness run on real audio. Nothing is hand-edited.</p>
  </div>
</div>
{note}
<section>
  <div class="sec-head">
    <p class="type-label">metrics</p>
    <h2>What we measure</h2>
  </div>
  <div class="card"><table class="defs">
<thead><tr><th style="text-align:left">Metric</th><th style="text-align:left">Definition</th></tr></thead>
<tbody>
<tr><td style="text-align:left">WER / CER</td><td style="text-align:left">Word error rate against a reference transcript (edit distance on word sequences). Languages without whitespace-delimited words (ja, zh, ko, th) are scored with character error rate instead - word error rate is meaningless there.</td></tr>
<tr><td style="text-align:left">p50 latency</td><td style="text-align:left">Median wall-clock time per transcription call, in milliseconds.</td></tr>
<tr><td style="text-align:left">Cost/min</td><td style="text-align:left">Effective USD per audio minute, from the provider's metered pricing over the audio actually processed.</td></tr>
<tr><td style="text-align:left">Score</td><td style="text-align:left">1 - WER when reference transcripts exist, otherwise inverse latency. This is the score the router's <code>benchmark</code> strategy routes on.</td></tr>
</tbody></table>
</div>
</section>
<section>
  <div class="sec-head">
    <p class="type-label">harness</p>
    <h2>How samples are run</h2>
  </div>
  <div class="card card-pad prose">
<p>Samples are short <code>.wav</code> clips per language in <code>benchmarks/samples/</code>, with same-named reference transcripts in <code>benchmarks/references/</code>. The harness (<code>benchmarks/harness.py</code>) sends every sample to every provider that supports the language, in the same process, back to back:</p>
<pre>python -m benchmarks.harness --samples benchmarks/samples \\
    --references benchmarks/references --language en</pre>
<p>Results land in <code>benchmarks/results.json</code>; this site is regenerated from that file. Nothing is hand-edited: if a number is on the leaderboard, it came out of a harness run.</p>
  </div>
</section>
<section>
  <div class="sec-head">
    <p class="type-label">audio</p>
    <h2>Where the audio comes from</h2>
  </div>
  <div class="card card-pad prose">
<p>The sample set is real human speech from <a href="https://huggingface.co/datasets/google/fleurs" target="_blank" rel="noreferrer">FLEURS</a> (CC-BY-4.0) - Google's Few-shot Learning Evaluation of Universal Representations of Speech corpus (<a href="https://arxiv.org/abs/2205.12446" target="_blank" rel="noreferrer">paper</a>). FLEURS is the standard open benchmark for multilingual speech: 102 languages, roughly 12 hours of speech per language, n-way parallel sentences built on the FLoRes-101 translation set. It is the evaluation OpenAI's Whisper paper reported its multilingual numbers on, which is what makes it the common ruler here.</p>
<p>We use 15 test-split clips per language for English (en_us), Spanish (es_419), Hindi (hi_in), French (fr_fr), German (de_de) and Japanese (ja_jp) - 90 clips per provider, 540 runs per full board. Clips are 4-14 seconds, picked at even intervals across the test split for speaker diversity, normalized to 16 kHz mono PCM. Reference transcripts are FLEURS' normalized transcriptions. No synthetic or TTS-generated audio. Clips and references are committed in <code>benchmarks/samples/</code> and <code>benchmarks/references/</code> so anyone can rerun the exact matrix.</p>
<p>Know the limits: FLEURS is clean, read speech - single speakers, quiet conditions, short utterances. That makes it the right common ruler, but it does not measure telephony noise, heavy accents, crosstalk or bad microphones. Every provider scores worse on real call audio than on this board. Treat these numbers as a ranking, not a promise.</p>
  </div>
</section>
<section>
  <div class="sec-head">
    <p class="type-label">cadence</p>
    <h2>Refresh cadence</h2>
  </div>
  <div class="card card-pad prose">
<p>A GitHub Action regenerates the site on every push that touches <code>benchmarks/</code> and on a weekly schedule, because provider quality drifts. The leaderboard links the exact Action run that produced its current numbers. Anyone can reproduce them: clone the repo, add provider keys, run the harness.</p>
  </div>
</section>
<section>
  <div class="sec-head">
    <p class="type-label">honesty</p>
    <h2>Sample data policy</h2>
  </div>
  <div class="card card-pad prose">
<p>Before the first real multi-provider run, the site renders <code>benchmarks/sample_results.json</code> - synthetic numbers that exist only to demonstrate the format. Sample data is always shown behind a visible SAMPLE DATA banner and is never presented as a real measurement. The moment a real run lands in <code>benchmarks/results.json</code>, it replaces the sample data automatically.</p>
  </div>
</section>
"""
    return page("Methodology", "methodology", body, generated_at)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                    help="harness results JSON (default: benchmarks/results.json)")
    ap.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE,
                    help="fallback sample data (default: benchmarks/sample_results.json)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output directory for the static site (default: docs/)")
    args = ap.parse_args()

    source = args.results
    payload = load_payload(args.results) if args.results.exists() else {}
    if not has_real_runs(payload):
        if args.sample.exists():
            print(f"no real provider runs in {args.results} - using sample data {args.sample}")
            payload = load_payload(args.sample)
            source = args.sample
        elif not payload.get("runs"):
            raise SystemExit(f"no results at {args.results} and no sample data at {args.sample}")

    sample = bool(payload.get("sample_data"))
    rows = aggregate(payload)
    tts_rows = aggregate_tts(payload)
    llm_rows = aggregate_llm(payload)
    run_url = payload.get("run_url")
    generated_at = payload.get("generated_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.html").write_text(
        render_index(rows, sample, generated_at, source.name, run_url, tts_rows, llm_rows))
    (args.out / "methodology.html").write_text(render_methodology(sample, generated_at, run_url))

    # Machine-readable surface: the same numbers, as JSON, served with the site.
    data_dir = args.out / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "results.json").write_text(json.dumps(payload, indent=2))
    board = {
        "generated_at": generated_at,
        "run_url": run_url,
        "dataset": {
            "name": "FLEURS", "license": "CC-BY-4.0",
            "url": "https://huggingface.co/datasets/google/fleurs",
            "clips_per_language": 15, "languages": sorted({l for r in rows for l in r["langs"]}),
        },
        "stt": rows, "tts": tts_rows, "llm": llm_rows,
    }
    (data_dir / "leaderboard.json").write_text(json.dumps(board, indent=2, default=str))
    print(f"wrote {args.out}/index.html, methodology.html and data/ "
          f"({len(rows)} stt rows, {len(tts_rows)} tts rows, {len(llm_rows)} llm rows, sample_data={sample})")


if __name__ == "__main__":
    main()
