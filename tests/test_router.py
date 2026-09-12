import time

import pytest

from voice_router.config import load_config
from voice_router.models import CallRecord
from voice_router.providers.registry import build_providers
from voice_router.router import RoutingEngine
from voice_router.telemetry import Telemetry


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEMETRY_PATH", str(tmp_path / "t.jsonl"))
    # dummy keys so every configured provider registers as available
    for k in ("DEEPGRAM_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY",
              "ELEVENLABS_API_KEY", "CARTESIA_API_KEY"):
        monkeypatch.setenv(k, "test-key")
    cfg = load_config()
    tel = Telemetry()
    return RoutingEngine(cfg, tel, build_providers(cfg))


def test_cost_strategy_orders_cheapest_first(engine):
    choices = engine.rank("stt", "en", "cost", units=10)
    costs = [c.estimated_cost_usd for c in choices]
    assert costs == sorted(costs)
    assert costs[0] < costs[-1]


def test_latency_strategy_uses_telemetry_over_defaults(engine):
    engine.telemetry.record(CallRecord(
        ts=time.time(), capability="stt", provider="openai-whisper",
        model="whisper-1", language="en", latency_ms=10,
        units=1, cost_usd=0.006, status="ok"))
    choices = engine.rank("stt", "en", "latency")
    assert choices[0].provider == "openai-whisper"


def test_language_filter_excludes_unsupported(engine):
    choices = engine.rank("stt", "xx", "cost")
    assert choices and all(c.provider.startswith("mock") for c in choices)


def test_pinned_model_beats_strategy(engine):
    choices = engine.resolve("stt", "deepgram:nova-3", "en", "cost", 1.0)
    assert len(choices) == 1
    assert choices[0].provider == "deepgram"
    assert choices[0].model == "nova-3"


def test_benchmark_strategy_prefers_scores(engine):
    engine._benchmarks = {"stt": {"openai-whisper": {"en": 0.95}, "deepgram": {"en": 0.80}}}
    choices = engine.rank("stt", "en", "benchmark")
    assert choices[0].provider == "openai-whisper"


def test_round_robin_rotates(engine):
    first = [c.provider for c in engine.rank("tts", "en", "round_robin")]
    second = [c.provider for c in engine.rank("tts", "en", "round_robin")]
    assert first != second and sorted(first) == sorted(second)
