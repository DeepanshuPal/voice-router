"""CPU-only end-of-turn benchmark for open and open-weight local models.

The corpus pairs each complete FLEURS utterance with a deterministic prefix of
that same clip. This makes the label boundary auditable and keeps API cost at
zero. It is a small regression board, not a claim about production calls.

Install the optional runner dependencies before use:
  pip install onnxruntime transformers huggingface-hub soundfile

Run:
  python -m benchmarks.turn_taking.run
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
import unicodedata
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = Path(__file__).with_name("corpus.json")
DEFAULT_RESULTS = Path(__file__).with_name("results.json")


@dataclass(frozen=True)
class Example:
    id: str
    language: str
    audio: np.ndarray
    text: str
    expected_end: bool
    kind: str


class Detector(Protocol):
    name: str
    model: str
    input_type: str
    languages: set[str]

    def predict(self, example: Example) -> tuple[bool, float]: ...


def read_wav(path: Path) -> np.ndarray:
    """Decode any WAV encoding to the model's 16 kHz mono float samples."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
         "-acodec", "pcm_f32le", "-ar", "16000", "-ac", "1", "-"],
        check=True, stdout=subprocess.PIPE,
    ).stdout
    audio = np.frombuffer(raw, dtype="<f4")
    if not len(audio):
        raise ValueError(f"{path} decoded to empty audio")
    return audio


def load_examples(path: Path) -> tuple[dict, list[Example]]:
    corpus = json.loads(path.read_text())
    ratio = float(corpus["split_ratio"])
    examples: list[Example] = []
    for item in corpus["samples"]:
        audio = read_wav(ROOT / item["audio"])
        text = (ROOT / item["reference"]).read_text().strip()
        words = text.split()
        cut_words = max(1, min(len(words) - 1, round(len(words) * ratio)))
        cut_audio = max(1, min(len(audio) - 1, round(len(audio) * ratio)))
        examples.extend([
            Example(item["id"] + "-complete", item["language"], audio,
                    text.rstrip(".?!।") + ("।" if item["language"] == "hi" else "."),
                    True, "complete"),
            Example(item["id"] + "-prefix", item["language"], audio[:cut_audio],
                    " ".join(words[:cut_words]), False, "derived-prefix"),
        ])
    return corpus, examples


class SmartTurn:
    name = "Pipecat"
    model = "Smart Turn v3.2 CPU"
    input_type = "audio"
    languages = {"en", "hi"}

    def __init__(self) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import WhisperFeatureExtractor
        model = hf_hub_download("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx")
        opts = ort.SessionOptions()
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = min(4, max(1, (os_cpu_count() + 1) // 2))
        self.session = ort.InferenceSession(model, sess_options=opts, providers=["CPUExecutionProvider"])
        self.extractor = WhisperFeatureExtractor(chunk_length=8)

    def predict(self, example: Example) -> tuple[bool, float]:
        n = 8 * 16000
        audio = example.audio[-n:]
        if len(audio) < n:
            audio = np.pad(audio, (n - len(audio), 0))
        inputs = self.extractor(audio, sampling_rate=16000, return_tensors="np",
                                padding="max_length", max_length=n, truncation=True,
                                do_normalize=True)
        features = inputs.input_features.astype(np.float32)
        prob = float(self.session.run(None, {"input_features": features})[0].flatten()[0])
        return prob > 0.5, prob


def os_cpu_count() -> int:
    import os
    return os.cpu_count() or 1


class LiveKitTurn:
    name = "LiveKit"
    input_type = "text"

    def __init__(self, variant: str) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
        if variant not in {"intl", "en"}:
            raise ValueError(variant)
        self.variant = variant
        revision = "v0.4.1-intl" if variant == "intl" else "v1.2.2-en"
        self.model = "turn-detector " + revision
        self.languages = {"en", "hi"} if variant == "intl" else {"en"}
        onnx = hf_hub_download("livekit/turn-detector", "onnx/model_q8.onnx", revision=revision)
        lang_path = hf_hub_download("livekit/turn-detector", "languages.json", revision=revision)
        self.thresholds = {k: v["threshold"] for k, v in json.loads(Path(lang_path).read_text()).items()}
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(4, max(1, math.ceil(os_cpu_count() / 2)))
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"], sess_options=opts)
        self.tokenizer = AutoTokenizer.from_pretrained("livekit/turn-detector", revision=revision,
                                                       truncation_side="left")

    def _text(self, raw: str) -> str:
        if self.variant == "intl":
            raw = unicodedata.normalize("NFKC", raw.lower())
            raw = "".join(c for c in raw if not (unicodedata.category(c).startswith("P") and c not in "'-"))
            raw = re.sub(r"\s+", " ", raw).strip()
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": raw}], add_generation_prompt=False,
            add_special_tokens=False, tokenize=False)
        return rendered[: rendered.rfind("<|im_end|>")]

    def predict(self, example: Example) -> tuple[bool, float]:
        inputs = self.tokenizer(self._text(example.text), add_special_tokens=False,
                                return_tensors="np", max_length=128, truncation=True)
        prob = float(self.session.run(None, {"input_ids": inputs["input_ids"].astype("int64")})[0].flatten()[-1])
        threshold = self.thresholds[example.language if self.variant == "intl" else "en"]
        return prob >= threshold, prob


class TurnSense:
    name = "TurnSense"
    model = "SmolLM2-135M quantized"
    input_type = "text"
    languages = {"en"}

    def __init__(self) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
        path = hf_hub_download("latishab/turnsense", "model_quantized.onnx")
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained("latishab/turnsense")

    def predict(self, example: Example) -> tuple[bool, float]:
        inputs = self.tokenizer(f"<|user|> {example.text}", padding="max_length",
                                max_length=256, return_tensors="np", truncation=True)
        output = self.session.run(None, {
            "input_ids": inputs["input_ids"].astype("int64"),
            "attention_mask": inputs["attention_mask"].astype("int64"),
        })[0][0]
        shifted = output - np.max(output)
        probs = np.exp(shifted) / np.exp(shifted).sum()
        return int(np.argmax(output)) == 1, float(probs[1])


def score(detector: Detector, examples: list[Example]) -> dict:
    records = []
    # One warm-up removes model initialization and first-call graph setup from latency.
    eligible = [e for e in examples if e.language in detector.languages]
    detector.predict(eligible[0])
    for example in eligible:
        started = time.perf_counter()
        predicted, probability = detector.predict(example)
        elapsed_ms = (time.perf_counter() - started) * 1000
        records.append({
            "id": example.id, "language": example.language, "kind": example.kind,
            "expected_end": example.expected_end, "predicted_end": predicted,
            "probability": round(probability, 6), "inference_ms": round(elapsed_ms, 3),
        })
    tp = sum(r["expected_end"] and r["predicted_end"] for r in records)
    tn = sum(not r["expected_end"] and not r["predicted_end"] for r in records)
    fp = sum(not r["expected_end"] and r["predicted_end"] for r in records)
    fn = sum(r["expected_end"] and not r["predicted_end"] for r in records)
    end_total, incomplete_total = tp + fn, tn + fp
    return {
        "provider": detector.name, "model": detector.model, "input_type": detector.input_type,
        "languages": sorted(detector.languages), "samples": len(records),
        "accuracy": round((tp + tn) / len(records), 4),
        "end_recall": round(tp / end_total, 4),
        "false_cutoff_rate": round(fp / incomplete_total, 4),
        "p50_inference_ms": round(statistics.median(r["inference_ms"] for r in records), 3),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}, "runs": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--models", default="smart-turn,livekit-en,livekit-intl,turnsense")
    args = parser.parse_args()
    corpus, examples = load_examples(args.corpus)
    constructors = {
        "smart-turn": SmartTurn,
        "livekit-en": lambda: LiveKitTurn("en"),
        "livekit-intl": lambda: LiveKitTurn("intl"),
        "turnsense": TurnSense,
    }
    results = []
    for name in args.models.split(","):
        print(f"loading {name}...", flush=True)
        detector = constructors[name]()
        row = score(detector, examples)
        print(f"  accuracy={row['accuracy']:.1%} p50={row['p50_inference_ms']:.1f}ms", flush=True)
        results.append(row)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample_data": False,
        "runtime": {"device": "CPU", "note": "One warm-up, then wall-clock inference per example."},
        "corpus": {k: v for k, v in corpus.items() if k != "samples"} | {"source_samples": len(corpus["samples"]), "paired_examples": len(examples)},
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
