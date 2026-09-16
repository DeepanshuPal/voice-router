# Accuracy is not latency: routing lessons from the corrected run

The corrected 16 September run makes the reason for a router concrete. There is no single STT winner. The best choice changes with the cost of a mistake and the cost of waiting.

Across the six-language FLEURS slice, AssemblyAI Universal-3.5 Pro had the highest mean accuracy score at **0.963**, but its median request-to-completion time was about **3.06 s**. Groq Whisper averaged **0.936** at about **407 ms**. In other words, the accuracy-first choice took roughly 7.5x as long as the low-latency choice in this batch measurement.

Speechmatics is the other quality-first option: **0.959** mean score at about **3.44 s**. Deepgram Nova-3 sat closer to the realtime side at **0.931** and **529 ms**. These are not interchangeable protocols: AssemblyAI and Speechmatics are asynchronous batch APIs, while Groq and Deepgram Nova are synchronous batch calls. Deepgram Flux is measured separately as a paced WebSocket stream.

## What the router should do

- **Accuracy strategy:** AssemblyAI first, Speechmatics as fallback. Good for transcripts, notes and workflows where a wrong word costs more than a few seconds.
- **Latency strategy:** Groq first, then Deepgram/Smallest based on live telemetry. Good for interactive turns where dead air is the main failure.
- **Language-aware quality:** do not use one global order. Speechmatics led Hindi among the fully completed providers; AssemblyAI led English, German and Japanese; Groq and Gladia tied on the current Spanish score.
- **Health-aware fallback:** benchmark rank is only the initial policy. Current failures and observed latency must be allowed to reorder or skip providers.

The router already exposes `benchmark`, `latency`, `cost`, and `round_robin` strategies. The next policy step is a constrained strategy such as "best accuracy under 800 ms," rather than forcing callers to choose one axis.

## Reproducibility note

Figures above are calculated from `benchmarks/results.json` committed by run [35074596486](https://github.com/DeepanshuPal/voice-router/actions/runs/35074596486). They use the corrected normalization, corpus-level scoring, serial execution, adaptive polling and protocol labels. They are an engineering analysis of that artifact, not a claim that one short corpus predicts every production workload.
