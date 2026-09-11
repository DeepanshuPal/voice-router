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
- Hosted leaderboard page generated from `results.json`

## License

MIT
