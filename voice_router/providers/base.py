"""Adapter contracts. Implement these two methods to add a provider."""

from __future__ import annotations

from ..config import ProviderSpec
from dataclasses import dataclass
import time


@dataclass(frozen=True)
class TTSTiming:
    audio: bytes
    protocol: str
    ttfb_ms: float | None
    completion_ms: float



class ProviderError(Exception):
    """The provider was reachable but failed the call."""


class ProviderUnavailable(ProviderError):
    """The provider can't serve this call at all (no key, bad language)."""


class STTProvider:
    def __init__(self, spec: ProviderSpec):
        self.spec = spec
        self.name = spec.name

    def supports(self, language: str) -> bool:
        return "*" in self.spec.languages or language in self.spec.languages

    async def transcribe(self, audio: bytes, model: str, language: str) -> str:
        raise NotImplementedError


class TTSProvider:
    def __init__(self, spec: ProviderSpec):
        self.spec = spec
        self.name = spec.name

    def supports(self, language: str) -> bool:
        return "*" in self.spec.languages or language in self.spec.languages

    def model_for(self, language: str) -> str:
        """Model to use for a language; most providers run one multilingual model."""
        return self.spec.models[0]

    async def synthesize(self, text: str, model: str, voice: str) -> bytes:
        raise NotImplementedError

    async def synthesize_timed(self, text: str, model: str, voice: str) -> TTSTiming:
        """Default for batch-only or not-yet-stream-instrumented adapters."""
        started = time.perf_counter()
        audio = await self.synthesize(text, model, voice)
        elapsed = (time.perf_counter() - started) * 1000
        return TTSTiming(audio=audio, protocol="batch", ttfb_ms=None, completion_ms=elapsed)


def sine_wav(seconds: float = 0.6, freq: float = 440.0, rate: int = 16000) -> bytes:
    """A real, playable WAV - used by the mock TTS and the test suite."""
    import math
    import struct
    import wave
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / rate)))
            for i in range(int(seconds * rate))
        )
        w.writeframes(frames)
    return buf.getvalue()
