"""Render the turn-taking results as a standalone static benchmark page."""
from __future__ import annotations
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks/turn_taking/results.json"
OUT = ROOT / "docs/turn-taking"
REPO = "https://github.com/DeepanshuPal/voice-router"


def e(value: object) -> str:
    return html.escape(str(value))


def main() -> None:
    payload = json.loads(RESULTS.read_text())
    rows = payload["results"]
    body = []
    for row in sorted(rows, key=lambda r: (-r["accuracy"], r["p50_inference_ms"])):
        body.append(f'''<tr><td>{e(row["provider"])}</td><td><strong>{e(row["model"])}</strong><small>{e(row["input_type"])} · {e(", ".join(row["languages"]))}</small></td><td>{row["accuracy"]:.1%}</td><td>{row["end_recall"]:.1%}</td><td>{row["false_cutoff_rate"]:.1%}</td><td>{row["p50_inference_ms"]:.1f} ms</td><td>{row["samples"]}</td></tr>''')
    generated = payload["generated_at"][:10]
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Turn-taking benchmark · voice-router</title><meta name="description" content="Open CPU end-of-turn models measured on the same English and Hindi corpus."><style>
@font-face{{font-family:g;src:url(https://cdn.jsdelivr.net/npm/geist@1.3.1/dist/fonts/geist-sans/Geist-Variable.woff2)}}@font-face{{font-family:m;src:url(https://cdn.jsdelivr.net/npm/geist@1.3.1/dist/fonts/geist-mono/GeistMono-Variable.woff2)}}*{{box-sizing:border-box}}body{{margin:0;color:#111;font-family:g,system-ui;background:#fff}}a{{color:inherit}}nav{{height:57px;border-bottom:1px solid #e9e9e9;display:flex;align-items:center}}.wrap{{max-width:1152px;margin:auto;padding:0 24px;width:100%}}nav .wrap{{display:flex;justify-content:space-between;align-items:center}}nav a{{text-decoration:none;font-size:13px;color:#555;margin-left:20px}}.brand{{font-size:15px!important;font-weight:550;color:#111!important;margin:0!important}}main{{padding-top:64px;padding-bottom:80px}}.eyebrow{{font:11px m,monospace;text-transform:uppercase;letter-spacing:.14em;color:#777}}h1{{font-size:38px;line-height:1.12;letter-spacing:-.035em;font-weight:520;max-width:760px;margin:14px 0}}.lede{{font-size:16px;line-height:1.65;color:#666;max-width:720px}}.meta{{display:flex;gap:28px;margin-top:34px;font:12px m,monospace;color:#777}}section{{margin-top:64px}}h2{{font-size:21px;font-weight:520;margin:8px 0 10px}}.sub{{font-size:13px;color:#777;max-width:690px;line-height:1.6}}.card{{border:1px solid #e4e4e4;border-radius:12px;overflow:auto;margin-top:20px}}table{{width:100%;border-collapse:collapse;font-size:13.5px}}th,td{{padding:14px 16px;border-bottom:1px solid #eee;text-align:right;white-space:nowrap}}th{{font:10.5px m,monospace;text-transform:uppercase;letter-spacing:.1em;color:#777;font-weight:400}}th:first-child,th:nth-child(2),td:first-child,td:nth-child(2){{text-align:left}}tr:last-child td{{border-bottom:0}}td{{font-family:m,monospace;font-variant-numeric:tabular-nums}}td:first-child{{font-family:g,system-ui;font-weight:550}}td:nth-child(2){{font-family:g,system-ui}}small{{display:block;color:#888;margin-top:3px;font:11px m,monospace}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:20px}}.box{{border:1px solid #e4e4e4;border-radius:12px;padding:20px}}.box strong{{font:13px m,monospace}}.box p{{font-size:13px;color:#666;line-height:1.55;margin:10px 0 0}}.note{{background:#f7f7f8;border-radius:12px;padding:22px;font-size:13px;color:#555;line-height:1.7;margin-top:20px}}footer{{border-top:1px solid #eee;padding:28px 0;color:#777;font-size:12px}}@media(max-width:750px){{h1{{font-size:31px}}.grid{{grid-template-columns:1fr}}nav a.hide{{display:none}}.meta{{display:block;line-height:2}}}}
</style></head><body><nav><div class="wrap"><a class="brand" href="../">● voice-router</a><div><a class="hide" href="../">Leaderboard</a><a href="../arena/">Arena</a><a href="{REPO}">GitHub</a></div></div></nav><main class="wrap"><p class="eyebrow">Turn-taking · CPU benchmark</p><h1>When should the agent answer?</h1><p class="lede">Four local end-of-turn models, one auditable corpus, zero API cost. We measure the failure users feel most: cutting them off before they finish.</p><div class="meta"><span>RUN {e(generated)}</span><span>24 PAIRED EXAMPLES</span><span>ENGLISH + HINDI</span><span>CPU ONLY</span></div>
<section><p class="eyebrow">Results</p><h2>Same decision, different signals</h2><p class="sub">Accuracy and end recall are higher-is-better. False cutoff is lower-is-better. Latency is warm median wall time on this runner; compare models within this run, not across hardware.</p><div class="card"><table><thead><tr><th>Provider</th><th>Model</th><th>Accuracy</th><th>End recall</th><th>False cutoff</th><th>p50 inference</th><th>n</th></tr></thead><tbody>{''.join(body)}</tbody></table></div></section>
<section><p class="eyebrow">How to read it</p><div class="grid"><div class="box"><strong>END RECALL</strong><p>Of complete utterances, how many the model lets the agent answer after. A miss feels like dead air.</p></div><div class="box"><strong>FALSE CUTOFF</strong><p>Of deliberately incomplete prefixes, how many the model treats as finished. This is the interruption rate.</p></div><div class="box"><strong>INPUT TYPE</strong><p>Smart Turn reads audio and prosody. LiveKit and TurnSense read the reference transcript, so upstream STT latency is not included.</p></div></div></section>
<section><p class="eyebrow">Corpus and limits</p><h2>A small public regression set, not a production claim</h2><div class="note">The source is 12 real CC-BY-4.0 FLEURS clips: six English and six Hindi. Each full clip is a complete turn. Its deterministic 55% audio/text prefix is labeled incomplete, giving 24 balanced examples. These prefixes are derived test cases, not natural recorded interruptions. That makes the boundary easy to audit but favors reproducibility over ecological realism. Every prediction, probability, confusion count, model revision and runner lives in <a href="data/results.json">raw JSON</a> and the <a href="{REPO}/tree/main/benchmarks/turn_taking">repo</a>.</div></section></main><footer><div class="wrap">voice-router · measured, not marketed · <a href="../methodology.html">methodology</a></div></footer></body></html>'''
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(page)
    (OUT / "data").mkdir(exist_ok=True)
    (OUT / "data/results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
