"""Request/response schemas. STT and TTS mirror OpenAI's audio API shapes."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Capability = Literal["stt", "tts", "llm"]
Strategy = Literal["benchmark", "cost", "latency", "round_robin"]


class RouteRequest(BaseModel):
    capability: Capability
    language: str = "en"
    strategy: Optional[Strategy] = None  # falls back to config default
    units: float = 1.0  # minutes of audio for STT, kilo-chars for TTS


class RouteChoice(BaseModel):
    provider: str
    model: str
    estimated_cost_usd: float
    expected_latency_ms: float
    benchmark_score: Optional[float] = None
    reason: str


class RouteResponse(BaseModel):
    strategy: str
    choices: list[RouteChoice]  # ordered; first is the winner


class SpeechRequest(BaseModel):
    model: str = "auto"  # "auto" or "provider:model"
    voice: str = "alloy"
    input: str = Field(..., max_length=4096)
    language: str = "en"
    strategy: Optional[Strategy] = None
    response_format: str = "wav"


class TranscriptionResponse(BaseModel):
    text: str
    provider: str
    model: str
    language: str
    latency_ms: float
    cost_usd: float
    fallbacks_tried: list[str] = []


class CallRecord(BaseModel):
    ts: float
    capability: str
    provider: str
    model: str
    language: str
    latency_ms: float
    units: float
    cost_usd: float
    status: str  # ok | error
    detail: str = ""
