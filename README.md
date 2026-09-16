# voice-router

One API in front of every voice AI provider. Route each call to the best STT, TTS, or LLM for that request - by cost, latency, or benchmark score. Bring your own keys. No lock-in.

If you're building voice agents, you already know the problem: Deepgram for one customer, Whisper for another, ElevenLabs until the bill arrives. Every provider has a different API, a different price, and a different failure mode at 2am. voice-router sits in front of all of them so your agent code never cares.

```text
your agent ──▶ voice-router ──▶ STT: Deepgram / Whisper / Groq
                     │         ▶ TTS: ElevenLabs / OpenAI / Cartesia
                     │         ▶ LLM: anything LiteLLM supports
                     └─ picks the winner per call, per your strategy
```

## Why not just pick one provider

- Prices differ 10x across providers for the same call.
- The "best" model changes per language, per accent, per week.
- Provider outages are normal. Hard-coded providers are downtime.
- You can't A/B providers if swapping one takes a refactor.

## Quickstart

```bash
git clone https://github.com/DeepanshuPal/voice-router
cd voice-router
pip install -r requirements.txt
cp .env.example .env        # add whatever keys you have - none required to try it
uvicorn voice_router.main:app --reload
```

No keys? It runs anyway on built-in mock providers so you can exercise the full routing path end to end:

```bash
# dry-run a routing decision
curl -s localhost:8000/v1/route -H 'content-type: application/json' -d '{
  "capability": "stt", "language": "es", "strategy": "cost"
}'

# transcribe (OpenAI-compatible)
curl -s localhost:8000/v1/audio/transcriptions \
  -F file=@sample.wav -F model=auto -F language=es

# synthesize speech
curl -s localhost:8000/v1/audio/speech -H 'content-type: application/json' -d '{
  "model": "auto", "voice": "alloy", "input": "Routed to the cheapest provider that speaks Spanish."
}' --output speech.wav
```

## Routing strategies

Set per request (`"strategy": "..."`) or default in `config/providers.yaml`.

| strategy      | picks the provider with...                                  |
|---------------|-------------------------------------------------------------|
| `benchmark`   | best benchmark score for that language (default)            |
| `cost`        | lowest price for this request's units                       |
| `latency`     | lowest observed p50 latency                                 |
| `round_robin` | next healthy provider in rotation                           |

Pin a provider anytime with `"model": "deepgram:nova-3"` instead of `auto`. Every strategy returns an ordered fallback chain - if the winner errors or times out, the next candidate takes the call automatically. Telemetry (latency, cost, outcome) is recorded per call and feeds the `latency` strategy and `/v1/telemetry`.

## Providers

| capability | providers shipped                        | key env var           |
|------------|------------------------------------------|-----------------------|
| STT        | Deepgram, OpenAI Whisper, Groq Whisper   | `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY` |
| TTS        | ElevenLabs, OpenAI, Cartesia             | `ELEVENLABS_API_KEY`, `OPENAI_API_KEY`, `CARTESIA_API_KEY` |
| LLM        | anything LiteLLM routes (optional extra) | provider-specific     |

Adding a provider = one config block + one small class implementing `transcribe()` or `synthesize()`. See `voice_router/providers/stt.py` for the pattern.

## Benchmarks: know, don't guess

The router is only as smart as its data. The harness runs real audio through every configured STT provider and records latency, cost, and (if you pass reference transcripts) word error rate:

```bash
python -m benchmarks.harness --samples benchmarks/samples --references benchmarks/references
```

Results land in `benchmarks/results.json` and power the `benchmark` strategy. Re-run monthly - provider quality drifts.

### How the numbers are made

Every number on the public board traces back to a file in this repo:

1. `benchmarks/harness.py` pushes real audio through every configured provider - 15 test-split clips per language from Google's [FLEURS](https://huggingface.co/datasets/google/fleurs) corpus, committed in `benchmarks/samples/` with reference transcripts in `benchmarks/references/` - and records WER (CER for ja/zh/ko/th), p50 latency and metered cost per clip.
2. `scripts/run_matrix.py` runs that matrix for every language and merges it into `benchmarks/results.json`. Nothing is hand-edited; when a provider can't run, the cell is empty instead of estimated.
3. A GitHub Action (`.github/workflows/leaderboard.yml`) re-runs the matrix on every `benchmarks/` push and weekly, then `scripts/generate_leaderboard.py` renders the site. The board links the exact Action run that produced its current numbers.

## Public leaderboard

Harness results are rendered into a static leaderboard: **https://deepanshupal.github.io/voice-router/** - WER, p50 latency, and effective cost per minute, per provider, per language, measured on 90 FLEURS clips per provider, with the [methodology](https://deepanshupal.github.io/voice-router/methodology.html) published alongside. Until the first real multi-provider run, it shows clearly-labeled sample data so the format is visible.

The site is plain HTML in `docs/`, served by GitHub Pages from the `main` branch - no build step and no separate `gh-pages` branch to drift away from the data that generates it. Regenerate after any harness run (a GitHub Action also does this on every `benchmarks/` push and weekly):

```bash
python scripts/generate_leaderboard.py
```

## Use it with Pipecat

Point Pipecat's OpenAI-compatible services at the router and your existing agent inherits routing with zero pipeline changes - see `examples/pipecat_agent.py`.

## API surface

```
POST /v1/audio/transcriptions   OpenAI-compatible STT
POST /v1/audio/speech           OpenAI-compatible TTS
POST /v1/chat/completions       LLM via LiteLLM (optional)
POST /v1/route                  dry-run a routing decision
GET  /v1/models                 virtual + provider models
GET  /v1/providers              status, pricing, benchmark scores
GET  /v1/telemetry              per-call latency/cost/outcome stats
```

## Roadmap

- WebSocket realtime endpoint (streaming STT/TTS in one socket)
- Telephony leg (Twilio / Plivo / Exotel)
- Usage metering + per-key budgets

## License

MIT

## Architecture

Voice Router is a realtime local-sidecar routing data plane; the public benchmark supplies policy evidence. See [the architecture decision](docs/architecture.md).


<!-- Refresh repository contributor metadata. -->
