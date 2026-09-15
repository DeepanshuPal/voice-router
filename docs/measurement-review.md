# Adversarial measurement review gate

**Rule:** no performance number reaches a public page until a reviewer whose only task is to invalidate the measurement signs off on the exact result artifact.

## Required attacks

### Accuracy

- Punctuation, casing, whitespace, Unicode and common contraction equivalence.
- Spoken-number versus digit equivalence for English.
- Corpus WER recomputed independently from stored reference and hypothesis.
- Demonstrate that short clips are weighted by reference units, not utterance count.
- Confirm the same normalizer and pinned version run on both sides.
- Confirm CER languages do not use whitespace WER.
- Inspect at least ten largest-error hypotheses per provider for language/script or empty-output failure.

### STT timing

- Confirm execution is serial and each clip has at least five repeats.
- Confirm runner region, timestamp, provider protocol and polling schedule exist in the artifact.
- Recompute median, Q1, Q3 and IQR from raw repeats.
- Reject any ranking that mixes streaming, synchronous batch and asynchronous batch.
- For streaming: validate timestamps for connection, first partial, stable partial and final.
- Confirm polling time is visible for async jobs rather than attributed to model compute.

### TTS timing

- Confirm streaming TTFB is timestamped at the first non-empty audio body chunk, not response completion.
- Confirm batch-only values are labeled exactly `full-synthesis wall clock (batch)`.
- Reject any claim that batch completion predicts conversational latency.
- Recompute median and IQR from at least five repeats.

### Publication

- Diff public HTML and JSON against the reviewed artifact.
- Verify the deployed site, not only generated files.
- Confirm methodology changelog describes any changed prior claim.
- Record reviewer, artifact commit, review date, attacks attempted and disposition.

## Implementation status

The corrected harness pieces are implemented but have not produced a public artifact:

- Batch STT runs serially with five repeats, records runner region, and keeps synchronous and asynchronous protocols separate.
- Streaming STT has its own event contract and harness. Deepgram Flux is the first implementation, using Listen v2 WebSocket, realtime-paced 80 ms PCM chunks, and client-observed connection, partial, stable-partial, final, and completion timestamps.
- TTS records first non-empty audio chunk separately from completion and labels batch completion independently.
- No provider run, result review, or publication approval has occurred. The current disposition below still controls.

Deepgram protocol references checked on 15 September 2026: [Flux quickstart](https://developers.deepgram.com/docs/flux/quickstart), [Listen v2 reference](https://developers.deepgram.com/reference/speech-to-text/listen-flux), and [CloseStream](https://developers.deepgram.com/docs/flux/close-stream).

## Current disposition

**15 September 2026: REJECTED / WITHDRAWN.** Historical STT/TTS/LLM accuracy and latency artifacts fail the gate because provider hypotheses, normalization parity, protocol-aware repeated timing and a pinned runner region were absent. The 24-example derived-prefix turn-taking board also fails ecological-validity review and remains offline.
