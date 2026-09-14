# Architecture decision: realtime routing data plane

**Decision date:** 15 September 2026  
**Decision:** Voice Router is a **realtime routing data plane**, implemented as a local sidecar. It is not a batch benchmark product with a BYOK dashboard.

## Why this is the product

Voice routing is paid on every conversational turn. A remote proxy adds another media-path hop, so routing must sit next to the agent runtime and keep audio streaming end to end. The public benchmark remains the evidence and policy-input system, but it is not the product architecture.

The sidecar accepts streaming audio over WebSocket, holds warm provider sessions, and selects a route before or at session start from a prefetched plan. It does not wait for a batch benchmark call while a user is speaking. Runtime fallbacks operate on already-open or immediately available streams.

## Data path

```text
agent runtime
  <-> local voice-router sidecar
        <-> streaming STT provider
        <-> turn detector
        <-> LLM endpoint
        <-> streaming TTS provider
```

The control service is outside the media path:

```text
benchmark/evaluation service -> signed routing plan -> local sidecar cache
local sidecar -> de-identified telemetry -> evaluation service
```

A routing plan contains provider/model order, language and task constraints, budget limits, health TTLs, and fallback rules. The sidecar prefetches it and continues from cache during control-service failure.

## Benchmark consequences

Streaming-native providers are the primary benchmark citizens. Deepgram Flux must receive a first-class row. The measurement boundary is protocol-aware:

- streaming STT: connection setup, first partial, stable partial, final transcript, realtime factor;
- streaming TTS: connection setup, time to first audio byte, audio chunk cadence, completion;
- async batch STT: job submit, provider processing, polling overhead, completion, labeled `async batch`;
- sync batch STT/TTS: request-to-completion, labeled `sync batch`.

These classes are not collapsed into one latency ranking. Accuracy can share a corpus when output semantics are comparable. Latency cannot.

## Deployment boundary

The open repository owns the sidecar protocol, adapters, benchmark harness, schemas, and reproducible public datasets. A future hosted service owns private evaluation suites, routing-plan generation, regression alerts, organization controls, and policy history. BYOK remains the first commercial model, but the keys are consumed by the local data plane rather than a hosted media proxy.

## What this rules out

- A hosted central gateway as the default audio path.
- A dashboard-only product whose router exists only on paper.
- Batch-only adapters defining the provider interface.
- One blended latency leaderboard across streaming, synchronous batch, and asynchronous jobs.
- Publishing a route recommendation without a reproducible measurement and adversarial review.

## Next implementation boundary

Define a streaming provider contract and event schema first, then build the measurement harness around it. Batch adapters become compatibility wrappers with explicit protocol labels. No further provider breadth work starts until the corrected scoring, protocol-aware latency harness, TTS TTFB instrumentation, correction log, and adversarial review gate are complete.
