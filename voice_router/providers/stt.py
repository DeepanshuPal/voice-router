"""STT adapters: Deepgram, OpenAI Whisper, Groq Whisper, and a zero-key mock."""

from __future__ import annotations

import os

import httpx

from .base import ProviderError, ProviderUnavailable, STTProvider


class DeepgramSTT(STTProvider):
    BASE = "https://api.deepgram.com/v1/listen"

    async def transcribe(self, audio: bytes, model: str, language: str) -> str:
        key = os.environ.get(self.spec.env_key)
        if not key:
            raise ProviderUnavailable("DEEPGRAM_API_KEY not set")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                self.BASE,
                params={"model": model, "language": language},
                headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"},
                content=audio,
            )
        if resp.status_code != 200:
            raise ProviderError(f"deepgram {resp.status_code}: {resp.text[:200]}")
        return resp.json()["results"]["channels"][0]["alternatives"][0]["transcript"]


class OpenAISTT(STTProvider):
    BASE = "https://api.openai.com/v1/audio/transcriptions"

    async def transcribe(self, audio: bytes, model: str, language: str) -> str:
        key = os.environ.get(self.spec.env_key)
        if not key:
            raise ProviderUnavailable("OPENAI_API_KEY not set")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                self.BASE,
                headers={"Authorization": f"Bearer {key}"},
                data={"model": model, "language": language},
                files={"file": ("audio.wav", audio, "audio/wav")},
            )
        if resp.status_code != 200:
            raise ProviderError(f"openai {resp.status_code}: {resp.text[:200]}")
        return resp.json()["text"]


class GroqSTT(STTProvider):
    BASE = "https://api.groq.com/openai/v1/audio/transcriptions"

    async def transcribe(self, audio: bytes, model: str, language: str) -> str:
        key = os.environ.get(self.spec.env_key)
        if not key:
            raise ProviderUnavailable("GROQ_API_KEY not set")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                self.BASE,
                headers={"Authorization": f"Bearer {key}"},
                data={"model": model, "language": language},
                files={"file": ("audio.wav", audio, "audio/wav")},
            )
        if resp.status_code != 200:
            raise ProviderError(f"groq {resp.status_code}: {resp.text[:200]}")
        return resp.json()["text"]


class MockSTT(STTProvider):
    """Always-on provider so the routing path runs end to end with no keys."""

    async def transcribe(self, audio: bytes, model: str, language: str) -> str:
        return f"[mock transcript: {len(audio)} bytes, language={language}]"


REGISTRY = {
    "deepgram": DeepgramSTT,
    "openai-whisper": OpenAISTT,
    "groq-whisper": GroqSTT,
    "mock-stt": MockSTT,
}
