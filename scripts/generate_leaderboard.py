"""Generate the static benchmark leaderboard site from harness results.

Reads benchmarks/results.json (produced by benchmarks/harness.py) and renders a
dependency-free static site into docs/ (served by GitHub Pages):

    python scripts/generate_leaderboard.py                 # results.json -> docs/
    python scripts/generate_leaderboard.py --results benchmarks/sample_results.json

If the results file has no real provider runs yet (empty, or only the built-in
mock providers), the generator falls back to benchmarks/sample_results.json so
the full loop renders end to end with zero provider keys. Sample data is always
rendered behind a visible SAMPLE DATA banner - it is never presented as a real
measurement.
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
    """One leaderboard row per (provider, language)."""
    rows: dict[tuple[str, str], dict] = {}
    scores = payload.get("scores", {}).get("stt", {})
    for run in payload.get("runs", []):
        if run.get("status") != "ok":
            continue
        key = (run["provider"], run.get("language") or "en")
        row = rows.setdefault(key, {
            "provider": run["provider"], "model": run.get("model", ""),
            "language": key[1], "samples": 0, "wers": [], "latencies": [],
            "cost": 0.0, "audio_s": 0.0,
        })
        row["samples"] += 1
        if run.get("wer") is not None:
            row["wers"].append(run["wer"])
        if run.get("latency_ms") is not None:
            row["latencies"].append(run["latency_ms"])
        row["cost"] += run.get("cost_usd") or 0.0
        row["audio_s"] += run.get("audio_s") or 0.0

    out = []
    for row in rows.values():
        row["avg_wer"] = (sum(row["wers"]) / len(row["wers"])) if row["wers"] else None
        row["p50_ms"] = statistics.median(row["latencies"]) if row["latencies"] else None
        row["cost_per_min"] = (row["cost"] / row["audio_s"] * 60) if row["audio_s"] else None
        row["score"] = scores.get(row["provider"], {}).get(row["language"])
        if row["score"] is None and row["avg_wer"] is not None:
            row["score"] = round(1 - row["avg_wer"], 4)
        out.append(row)
    out.sort(key=lambda r: (r["language"], -(r["score"] or 0)))
    return out


# ---------------------------------------------------------------- HTML helpers

CSS = """
:root{color-scheme:dark}
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#e6edf3;font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;line-height:1.55;padding:0 16px 64px}
a{color:#58a6ff;text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:960px;margin:0 auto}
header{padding:32px 0 8px}
h1{font-size:1.6rem;letter-spacing:-.01em}
h1 a{color:#e6edf3}
.sub{color:#8b949e;font-size:.95rem;margin-top:4px}
.banner{background:#3d2e00;border:1px solid #9e6a03;color:#f0b72f;border-radius:8px;padding:10px 14px;margin:16px 0;font-size:.9rem;font-weight:600}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:18px;margin-top:20px}
.card h2{font-size:1.05rem;margin-bottom:12px;color:#c9d1d9}
table{width:100%;border-collapse:collapse;font-size:.9rem}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid #21262d;white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
th{color:#8b949e;font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}
table.defs td{white-space:normal;vertical-align:top}
tr:last-child td{border-bottom:none}
.pill{display:inline-block;background:#1f6feb33;border:1px solid #1f6feb;color:#58a6ff;border-radius:999px;padding:1px 9px;font-size:.75rem;font-weight:600}
.best{color:#3fb950;font-weight:700}
.bar-row{display:flex;align-items:center;gap:10px;margin:6px 0;font-size:.85rem}
.bar-label{width:170px;color:#c9d1d9;flex:none;overflow:hidden;text-overflow:ellipsis}
.bar-track{flex:1;background:#21262d;border-radius:4px;height:16px;overflow:hidden}
.bar-fill{height:100%;border-radius:4px}
.bar-val{width:110px;text-align:right;color:#8b949e;flex:none;font-variant-numeric:tabular-nums}
footer{margin-top:32px;color:#8b949e;font-size:.8rem;border-top:1px solid #21262d;padding-top:16px}
nav{margin-top:8px;font-size:.85rem}
@media(max-width:640px){
  th:nth-child(3),td:nth-child(3){display:none}
  .bar-label{width:110px}.bar-val{width:84px;font-size:.78rem}
  .card{padding:14px}
}
"""

PALETTE = ["#58a6ff", "#3fb950", "#f0b72f", "#f778ba", "#76e3ea", "#d2a8ff", "#ffa657"]


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} - voice-router benchmarks</title>
<style>{CSS}</style>
</head>
<body><div class="wrap">
{body}
</div></body>
</html>
"""


def nav(active: str) -> str:
    lb = "<b>Leaderboard</b>" if active == "index" else '<a href="index.html">Leaderboard</a>'
    me = "<b>Methodology</b>" if active == "methodology" else '<a href="methodology.html">Methodology</a>'
    gh = '<a href="https://github.com/DeepanshuPal/voice-router">GitHub</a>'
    return f"<nav>{lb} &middot; {me} &middot; {gh}</nav>"


def fmt_wer(w) -> str:
    return "&mdash;" if w is None else f"{w * 100:.1f}%"


def fmt_cost(c) -> str:
    return "&mdash;" if c is None else f"${c:.4f}"


def render_table(rows: list[dict]) -> str:
    langs = sorted({r["language"] for r in rows})
    best_by_lang = {}
    for lang in langs:
        lang_rows = [r for r in rows if r["language"] == lang and r["score"] is not None]
        if lang_rows:
            best_by_lang[lang] = max(r["score"] for r in lang_rows)

    body_rows = []
    for r in rows:
        lang = r["language"]
        score = r["score"]
        cls = ' class="best"' if score is not None and score == best_by_lang.get(lang) else ""
        score_cell = "&mdash;" if score is None else f"{score:.3f}"
        p50 = r["p50_ms"]
        p50_cell = "&mdash;" if p50 is None else f"{p50:.0f} ms"
        body_rows.append(
            "<tr>"
            f"<td>{esc(r['provider'])}</td>"
            f"<td>{esc(r['model'])}</td>"
            f"<td><span class=\"pill\">{esc(lang)}</span></td>"
            f"<td{cls}>{score_cell}</td>"
            f"<td>{fmt_wer(r['avg_wer'])}</td>"
            f"<td>{p50_cell}</td>"
            f"<td>{fmt_cost(r['cost_per_min'])}</td>"
            f"<td>{r['samples']}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Provider</th><th>Model</th><th>Lang</th><th>Score</th>"
        "<th>WER</th><th>p50 latency</th><th>Cost/min</th><th>Samples</th>"
        "</tr></thead><tbody>" + "".join(body_rows) + "</tbody></table>"
    )


def bar_chart(title: str, items: list[tuple[str, float, str]], color_offset: int = 0) -> str:
    """items: (label, value, formatted value). Lower/higher agnostic - bar length
    is proportional to the max value."""
    if not items:
        return ""
    peak = max(v for _, v, _ in items) or 1.0
    rows = []
    for i, (label, value, shown) in enumerate(items):
        pct = max(2.0, value / peak * 100)
        color = PALETTE[(i + color_offset) % len(PALETTE)]
        rows.append(
            f'<div class="bar-row"><div class="bar-label">{esc(label)}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div></div>'
            f'<div class="bar-val">{esc(shown)}</div></div>'
        )
    return f'<div class="card"><h2>{esc(title)}</h2>{"".join(rows)}</div>'


def render_index(rows: list[dict], sample: bool, generated_at: str, source: str) -> str:
    langs = sorted({r["language"] for r in rows})
    primary = "en" if "en" in langs else (langs[0] if langs else "en")

    banner = ""
    if sample:
        banner = ('<div class="banner">SAMPLE DATA - these numbers are synthetic and exist '
                  "to demonstrate the format. They are not real measurements. "
                  'See <a href="methodology.html">methodology</a>.</div>')

    # Charts use the primary language, one row per provider.
    prim = [r for r in rows if r["language"] == primary]
    wer_items = sorted(
        [(f"{r['provider']} ({r['model']})", r["avg_wer"], fmt_wer(r["avg_wer"]).replace("&mdash;", "n/a"))
         for r in prim if r["avg_wer"] is not None],
        key=lambda x: x[1])
    lat_items = sorted(
        [(f"{r['provider']} ({r['model']})", r["p50_ms"], f"{r['p50_ms']:.0f} ms")
         for r in prim if r["p50_ms"] is not None],
        key=lambda x: x[1])

    lang_label = LANG_NAMES.get(primary, primary)
    body = f"""
<header>
  <h1><a href="https://github.com/DeepanshuPal/voice-router">voice-router</a> benchmarks</h1>
  <div class="sub">STT providers ranked by accuracy, latency, and cost - measured, not marketed.</div>
  {nav("index")}
</header>
{banner}
<div class="card"><h2>Leaderboard</h2>{render_table(rows)}</div>
{bar_chart(f"Word error rate - {lang_label} (lower is better)", wer_items)}
{bar_chart(f"p50 transcription latency - {lang_label} (lower is better)", lat_items, 2)}
<footer>
  Generated {esc(generated_at)} from <code>{esc(source)}</code> &middot;
  refresh: <code>python scripts/generate_leaderboard.py</code> &middot;
  how this is measured: <a href="methodology.html">methodology</a>
</footer>
"""
    return page("Leaderboard", body)


def render_methodology(sample: bool, generated_at: str) -> str:
    note = (
        '<div class="banner">The current leaderboard shows SAMPLE DATA - synthetic numbers '
        "demonstrating the format while real provider runs are being collected.</div>"
        if sample else ""
    )
    body = f"""
<header>
  <h1>Methodology</h1>
  <div class="sub">How the voice-router benchmark numbers are produced.</div>
  {nav("methodology")}
</header>
{note}
<div class="card"><h2>What we measure</h2>
<p style="margin-bottom:8px">Every run pushes the same audio samples through each configured STT provider and records, per sample:</p>
<table class="defs">
<thead><tr><th style="text-align:left">Metric</th><th style="text-align:left">Definition</th></tr></thead>
<tbody>
<tr><td style="text-align:left">WER</td><td style="text-align:left">Word error rate against a reference transcript (edit distance on word sequences). CER for character-based languages is on the roadmap.</td></tr>
<tr><td style="text-align:left">p50 latency</td><td style="text-align:left">Median wall-clock time per transcription call, in milliseconds.</td></tr>
<tr><td style="text-align:left">Cost/min</td><td style="text-align:left">Effective USD per audio minute, from the provider's metered pricing over the audio actually processed.</td></tr>
<tr><td style="text-align:left">Score</td><td style="text-align:left">1 - WER when reference transcripts exist, otherwise inverse latency. This is the score the router's <code>benchmark</code> strategy routes on.</td></tr>
</tbody></table>
</div>
<div class="card"><h2>How samples are run</h2>
<p style="margin-bottom:8px">Samples are short <code>.wav</code> clips per language in <code>benchmarks/samples/</code>, with same-named reference transcripts in <code>benchmarks/references/</code>. The harness (<code>benchmarks/harness.py</code>) sends every sample to every provider that supports the language, in the same process, back to back:</p>
<pre style="background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:12px;font-size:.85rem;overflow-x:auto">python -m benchmarks.harness --samples benchmarks/samples \\
    --references benchmarks/references --language en</pre>
<p style="margin-top:8px">Results land in <code>benchmarks/results.json</code>; this site is regenerated from that file. Nothing is hand-edited: if a number is on the leaderboard, it came out of a harness run.</p>
</div>
<div class="card"><h2>Refresh cadence</h2>
<p>A GitHub Action regenerates the site on every push that touches <code>benchmarks/</code> and on a weekly schedule, because provider quality drifts. Anyone can reproduce the numbers: clone the repo, add provider keys, run the harness.</p>
</div>
<div class="card"><h2>Sample data policy</h2>
<p>Before the first real multi-provider run, the site renders <code>benchmarks/sample_results.json</code> - synthetic numbers that exist only to demonstrate the format. Sample data is always shown behind a visible SAMPLE DATA banner and is never presented as a real measurement. The moment a real run lands in <code>benchmarks/results.json</code>, it replaces the sample data automatically.</p>
</div>
<footer>Generated {esc(generated_at)} &middot; <a href="index.html">back to the leaderboard</a></footer>
"""
    return page("Methodology", body)


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
    generated_at = payload.get("generated_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.html").write_text(render_index(rows, sample, generated_at, source.name))
    (args.out / "methodology.html").write_text(render_methodology(sample, generated_at))
    print(f"wrote {args.out}/index.html and {args.out}/methodology.html "
          f"({len(rows)} rows, sample_data={sample})")


if __name__ == "__main__":
    main()
