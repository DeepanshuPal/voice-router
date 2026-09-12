"""The routing engine: rank providers per request, walk the chain on failure."""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

from .config import ProviderSpec, RouterConfig
from .models import CallRecord, RouteChoice
from .providers.base import ProviderError
from .telemetry import Telemetry

RESULTS_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "results.json"


class RoutingEngine:
    def __init__(self, cfg: RouterConfig, telemetry: Telemetry, providers: dict):
        self.cfg = cfg
        self.telemetry = telemetry
        self.providers = providers  # {"stt": {name: adapter}, "tts": {...}}
        self._rr = itertools.count()
        self._benchmarks = self._load_benchmarks()

    @staticmethod
    def _load_benchmarks() -> dict:
        """benchmarks/results.json: {capability: {provider: {language: score}}}."""
        try:
            return json.loads(RESULTS_PATH.read_text()).get("scores", {})
        except (OSError, json.JSONDecodeError):
            return {}

    def candidates(self, capability: str, language: str) -> list[ProviderSpec]:
        specs = []
        for name, adapter in self.providers.get(capability, {}).items():
            if adapter.supports(language):
                specs.append(adapter.spec)
        return specs

    def rank(self, capability: str, language: str, strategy: str, units: float = 1.0) -> list[RouteChoice]:
        cands = self.candidates(capability, language)
        if not cands:
            return []

        def bench(spec: ProviderSpec) -> float:
            lang_scores = self._benchmarks.get(capability, {}).get(spec.name, {})
            return lang_scores.get(language, lang_scores.get("*", 0.0))

        def latency(spec: ProviderSpec) -> float:
            return self.telemetry.p50_latency(capability, spec.name) or spec.default_latency_ms

        choices = [
            RouteChoice(
                provider=s.name,
                model=s.models[0] if s.models else "default",
                estimated_cost_usd=round(s.unit_cost(units), 6),
                expected_latency_ms=latency(s),
                benchmark_score=bench(s) or None,
                reason="",
            )
            for s in cands
        ]

        if strategy == "cost":
            choices.sort(key=lambda c: c.estimated_cost_usd)
            reason = "lowest cost for these units"
        elif strategy == "latency":
            choices.sort(key=lambda c: c.expected_latency_ms)
            reason = "lowest observed p50 latency"
        elif strategy == "round_robin":
            n = next(self._rr) % len(choices)
            choices = choices[n:] + choices[:n]
            reason = "rotation"
        else:  # benchmark: quality first, latency breaks ties
            choices.sort(key=lambda c: (-(c.benchmark_score or 0.0), c.expected_latency_ms))
            reason = "best benchmark score for this language"

        for c in choices:
            c.reason = reason
        return choices

    def resolve(self, capability: str, model_field: str, language: str,
                strategy: str | None, units: float) -> list[RouteChoice]:
        """'provider:model' pins a single choice; 'auto' runs the strategy."""
        if model_field and model_field != "auto" and ":" in model_field:
            pname, pmodel = model_field.split(":", 1)
            spec = next((s for s in self.candidates(capability, language) if s.name == pname), None)
            if spec is None:
                return []
            return [RouteChoice(provider=pname, model=pmodel,
                                estimated_cost_usd=round(spec.unit_cost(units), 6),
                                expected_latency_ms=spec.default_latency_ms,
                                reason="pinned by request")]
        return self.rank(capability, language, strategy or self.cfg.default_strategy, units)

    async def call_stt(self, audio: bytes, language: str, model_field: str,
                       strategy: str | None) -> dict:
        minutes = max(len(audio) / (16000 * 2 * 60), 0.01)  # rough pcm16 estimate
        return await self._walk("stt", audio, language, model_field, strategy, minutes)

    async def call_tts(self, text: str, voice: str, language: str, model_field: str,
                       strategy: str | None) -> dict:
        kchars = max(len(text) / 1000.0, 0.001)
        return await self._walk("tts", text, language, model_field, strategy, kchars, voice=voice)

    async def _walk(self, capability, payload, language, model_field, strategy, units, voice=None):
        choices = self.resolve(capability, model_field, language, strategy, units)
        if not choices:
            raise ProviderError(f"no {capability} provider available for language '{language}'")

        tried, last_err = [], None
        chain = choices if self.cfg.fallback else choices[:1]
        for choice in chain:
            adapter = self.providers[capability][choice.provider]
            start = self.telemetry.now()
            try:
                if capability == "stt":
                    result = await adapter.transcribe(payload, choice.model, language)
                else:
                    result = await adapter.synthesize(payload, choice.model, voice or "alloy")
                latency = (self.telemetry.now() - start) * 1000
                self.telemetry.record(CallRecord(
                    ts=start, capability=capability, provider=choice.provider,
                    model=choice.model, language=language, latency_ms=latency,
                    units=units, cost_usd=choice.estimated_cost_usd, status="ok"))
                return {"result": result, "provider": choice.provider, "model": choice.model,
                        "latency_ms": round(latency, 1), "cost_usd": choice.estimated_cost_usd,
                        "fallbacks_tried": tried}
            except Exception as e:
                latency = (self.telemetry.now() - start) * 1000
                self.telemetry.record(CallRecord(
                    ts=start, capability=capability, provider=choice.provider,
                    model=choice.model, language=language, latency_ms=latency,
                    units=units, cost_usd=0.0, status="error", detail=str(e)[:200]))
                tried.append(f"{choice.provider}: {e}")
                last_err = e
        raise ProviderError(f"all {capability} providers failed; last: {last_err}")
