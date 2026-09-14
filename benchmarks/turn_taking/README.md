# Turn-taking benchmark

A zero-API-cost end-of-turn benchmark for four local CPU model variants:

- Pipecat Smart Turn v3.2 CPU, native audio, BSD-2-Clause
- LiveKit turn-detector v1.2.2 English, text, LiveKit Model License
- LiveKit turn-detector v0.4.1 International, text, LiveKit Model License
- TurnSense SmolLM2-135M quantized, text, Apache-2.0

The first corpus is deliberately small and auditable. It pairs 12 real FLEURS
utterances (6 English, 6 Hindi) with deterministic 55% prefixes. Full clips
are expected ends; prefixes are expected continuations. Prefixes are derived
evaluation examples, not claimed as natural recorded conversations.

Metrics are accuracy, recall on real ends, false-cutoff rate on incomplete
prefixes, and median warm inference time. Results from audio-native and
text-native models are shown together because both make the same pipeline
decision, but their input types stay visible.

```bash
python -m venv .venv-turn
. .venv-turn/bin/activate
pip install onnxruntime transformers huggingface-hub
python -m benchmarks.turn_taking.run
python scripts/generate_turn_taking.py
```

Model weights are fetched from their named Hugging Face repositories and are
not committed. `results.json` contains every per-example prediction.
