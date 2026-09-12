"""FastAPI surface. OpenAI-compatible audio endpoints plus routing introspection."""

from __future__ import annotations

import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from . import __version__
from .config import load_config
from .models import RouteRequest, RouteResponse, SpeechRequest, TranscriptionResponse
from .providers.base import ProviderError
from .providers.llm import chat_completion
from .providers.registry import build_providers
from .router import RoutingEngine
from .telemetry import Telemetry


def create_app(config_path: str | None = None) -> FastAPI:
    cfg = load_config(config_path)
    telemetry = Telemetry()
    engine = RoutingEngine(cfg, telemetry, build_providers(cfg))

    app = FastAPI(title="voice-router", version=__version__)
    app.state.engine = engine
    app.state.telemetry = telemetry

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "version": __version__}

    @app.get("/v1/models")
    def models():
        out = []
        for capability in ("stt", "tts"):
            for name, adapter in engine.providers[capability].items():
                for m in adapter.spec.models:
                    out.append({"id": f"{name}:{m}", "capability": capability, "provider": name})
        return {"object": "list", "data": out + [{"id": "auto", "capability": "any",
                                                  "provider": "routed"}]}

    @app.get("/v1/providers")
    def providers():
        out = {}
        for capability in ("stt", "tts"):
            for name, adapter in engine.providers[capability].items():
                s = adapter.spec
                out[name] = {
                    "capability": capability,
                    "models": s.models,
                    "languages": s.languages,
                    "cost_per_minute_usd": s.cost_per_minute_usd or None,
                    "cost_per_1k_chars_usd": s.cost_per_1k_chars_usd or None,
                    "p50_latency_ms": telemetry.p50_latency(capability, name),
                }
        return out

    @app.post("/v1/route", response_model=RouteResponse)
    def route(req: RouteRequest):
        choices = engine.resolve(req.capability, "auto", req.language, req.strategy, req.units)
        if not choices:
            raise HTTPException(404, f"no {req.capability} provider for language '{req.language}'")
        return RouteResponse(strategy=req.strategy or engine.cfg.default_strategy, choices=choices)

    @app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
    async def transcribe(
        file: UploadFile = File(...),
        model: str = Form("auto"),
        language: str = Form("en"),
        strategy: str | None = Form(None),
    ):
        audio = await file.read()
        if not audio:
            raise HTTPException(400, "empty audio file")
        try:
            r = await engine.call_stt(audio, language, model, strategy)
        except ProviderError as e:
            raise HTTPException(503, str(e))
        return TranscriptionResponse(text=r["result"], provider=r["provider"],
                                     model=r["model"], language=language,
                                     latency_ms=r["latency_ms"], cost_usd=r["cost_usd"],
                                     fallbacks_tried=r["fallbacks_tried"])

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest):
        try:
            r = await engine.call_tts(req.input, req.voice, req.language, req.model, req.strategy)
        except ProviderError as e:
            raise HTTPException(503, str(e))
        media = "audio/wav" if req.response_format == "wav" else "audio/mpeg"
        return Response(
            content=r["result"], media_type=media,
            headers={"x-voice-router-provider": r["provider"],
                     "x-voice-router-cost-usd": str(r["cost_usd"])})

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        models = body.get("models") or [body.get("model", "gpt-4o-mini")]
        try:
            return await chat_completion(body.get("messages", []), models)
        except ProviderError as e:
            raise HTTPException(503, str(e))

    @app.get("/v1/telemetry")
    def telemetry_summary():
        return telemetry.summary()

    return app


app = create_app(os.environ.get("ROUTER_CONFIG") or None)
