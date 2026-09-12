"""TTS adapters: ElevenLabs, OpenAI, Cartesia, and a zero-key mock."""

from __future__ import annotations

import os

import httpx

from .base import ProviderError, ProviderUnavailable, TTSProvider, sine_wav

# Default voices per provider when the caller passes an OpenAI-style name.
VOICE_MAP = {
    "elevenlabs": "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "openai-tts": "alloy",
    "cartesia": "79a125e8-cd45-4c13-8a67-188112f4c8",
}


class ElevenLabsTTS(TTSProvider):
    async def synthesize(self, text: str, model: str, voice: str) -> bytes:
        key = os.environ.get(self.spec.env_key)
        if not key:
            raise ProviderUnavailable("ELEVENLABS_API_KEY not set")
        voice_id = VOICE_MAP["elevenlabs"] if voice == "alloy" else voice
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={"xi-api-key": key, "Content-Type": "application/json"},
                json={"text": text, "model_id": model},
            )
        if resp.status_code != 200:
            raise ProviderError(f"elevenlabs {resp.status_code}: {resp.text[:200]}")
        return resp.content


class OpenAITTS(TTSProvider):
    async def synthesize(self, text: str, model: str, voice: str) -> bytes:
        key = os.environ.get(self.spec.env_key)
        if not key:
            raise ProviderUnavailable("OPENAI_API_KEY not set")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "voice": voice, "input": text},
            )
        if resp.status_code != 200:
            raise ProviderError(f"openai {resp.status_code}: {resp.text[:200]}")
        return resp.content


class CartesiaTTS(TTSProvider):
    async def synthesize(self, text: str, model: str, voice: str) -> bytes:
        key = os.environ.get(self.spec.env_key)
        if not key:
            raise ProviderUnavailable("CARTESIA_API_KEY not set")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.cartesia.ai/tts/bytes",
                headers={
                    "X-API-Key": key,
                    "Cartesia-Version": "2024-06-10",
                    "Content-Type": "application/json",
                },
                json={
                    "model_id": model,
                    "transcript": text,
                    "voice": {"mode": "id", "id": VOICE_MAP["cartesia"] if voice == "alloy" else voice},
                    "output_format": {"container": "wav", "encoding": "pcm_s16le", "sample_rate": 16000},
                },
            )
        if resp.status_code != 200:
            raise ProviderError(f"cartesia {resp.status_code}: {resp.text[:200]}")
        return resp.content


class MockTTS(TTSProvider):
    """Returns a playable sine-tone WAV so demos and tests need no keys."""

    async def synthesize(self, text: str, model: str, voice: str) -> bytes:
        seconds = min(2.0, max(0.3, len(text) / 80.0))
        return sine_wav(seconds=seconds)


REGISTRY = {
    "elevenlabs": ElevenLabsTTS,
    "openai-tts": OpenAITTS,
    "cartesia": CartesiaTTS,
    "mock-tts": MockTTS,
}
