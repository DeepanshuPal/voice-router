"""TTS adapters: ElevenLabs, OpenAI, Cartesia, Rime, Deepgram Aura, Groq Orpheus, Smallest Lightning, and a zero-key mock."""

from __future__ import annotations

import os

import httpx

from .base import ProviderError, ProviderUnavailable, TTSProvider, sine_wav

# Default voices per provider when the caller passes an OpenAI-style name.
VOICE_MAP = {
    "elevenlabs": "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "openai-tts": "alloy",
    "cartesia": "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
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


class DeepgramAuraTTS(TTSProvider):
    """Deepgram Aura-2: the voice is part of the model name (aura-2-<voice>-<lang>)."""

    LANG_VOICES = {
        "en": "aura-2-thalia-en",
        "es": "aura-2-agustina-es",
        "de": "aura-2-aurelia-de",
        "fr": "aura-2-agathe-fr",
        "ja": "aura-2-ama-ja",
    }

    def model_for(self, language: str) -> str:
        return self.LANG_VOICES.get(language, self.spec.models[0])

    async def synthesize(self, text: str, model: str, voice: str) -> bytes:
        key = os.environ.get(self.spec.env_key)
        if not key:
            raise ProviderUnavailable("DEEPGRAM_API_KEY not set")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.deepgram.com/v1/speak",
                params={"model": model},
                headers={"Authorization": f"Token {key}", "Content-Type": "application/json"},
                json={"text": text},
            )
        if resp.status_code != 200:
            raise ProviderError(f"deepgram-aura {resp.status_code}: {resp.text[:200]}")
        return resp.content


class GroqTTS(TTSProvider):
    """Groq Orpheus TTS over the OpenAI-compatible speech endpoint (English)."""

    async def synthesize(self, text: str, model: str, voice: str) -> bytes:
        key = os.environ.get(self.spec.env_key)
        if not key:
            raise ProviderUnavailable("GROQ_API_KEY not set")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/audio/speech",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "voice": "troy", "input": text,
                      "response_format": "wav"},
            )
        if resp.status_code != 200:
            raise ProviderError(f"groq-tts {resp.status_code}: {resp.text[:200]}")
        return resp.content


class RimeTTS(TTSProvider):
    """Rime: mistv2 for en/es/fr/de, coda for hi/ja.

    coda ignores audioFormat and streams raw PCM s16le 16 kHz back; wrap it in
    a WAV header so clips stay playable. mistv2 returns JSON with base64 audio.
    """

    BASE = "https://users.rime.ai/v1/rime-tts"
    LANG_MAP = {"en": "eng", "es": "spa", "fr": "fra", "de": "ger", "hi": "hin", "ja": "jpn"}
    # Voices are per (model, language) - astra only exists for mistv2/eng.
    # From https://users.rime.ai/data/voices/all-v2.json (2026-09-14).
    SPEAKER = {
        "mistv2": {"eng": "abbie", "spa": "diego", "fra": "alois", "ger": "amalia"},
        "coda": {"eng": "adeline", "hin": "nadi", "jpn": "akari"},
    }

    def model_for(self, language: str) -> str:
        # The language rides in the model string ("coda-hin") because the
        # synthesize() contract does not pass language separately.
        base = "coda" if language in ("hi", "ja") else "mistv2"
        return f"{base}-{self.LANG_MAP.get(language, 'eng')}"

    async def synthesize(self, text: str, model: str, voice: str) -> bytes:
        key = os.environ.get(self.spec.env_key)
        if not key:
            raise ProviderUnavailable("RIME_API_KEY not set")
        base, _, lang = model.rpartition("-")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                self.BASE,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"speaker": self.SPEAKER.get(base, {}).get(lang, "abbie"), "text": text,
                      "modelId": base, "lang": lang,
                      "audioFormat": "mp3", "sampleRate": 16000},
            )
        if resp.status_code != 200:
            raise ProviderError(f"rime {resp.status_code}: {resp.text[:200]}")
        if "json" in resp.headers.get("content-type", ""):
            import base64
            return base64.b64decode(resp.json()["audioContent"])
        import io, wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(resp.content)
        return buf.getvalue()


class SmallestTTS(TTSProvider):
    """Smallest AI Lightning v3.1 Pro: per-language curated voices.

    The standard v3.1 pool region-gates languages; the Pro pool serves
    en/es/de/fr/hi/ja on a free account. Voice IDs from
    docs.smallest.ai/.../voices-languages (2026-09-14).
    """

    BASE = "https://api.smallest.ai/waves/v1/tts"
    VOICE = {"en": "kaitlyn", "es": "martina", "de": "hanna",
             "fr": "manon", "hi": "meher", "ja": "aria"}

    def model_for(self, language: str) -> str:
        # Language rides in the model string: "lightning_v3.1_pro-de".
        return f"{self.spec.models[0]}-{language}"

    async def synthesize(self, text: str, model: str, voice: str) -> bytes:
        key = os.environ.get(self.spec.env_key)
        if not key:
            raise ProviderUnavailable("SMALLEST_API_KEY not set")
        base, _, lang = model.rpartition("-")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                self.BASE,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json",
                         "Accept": "audio/wav"},
                json={"text": text, "voice_id": self.VOICE.get(lang, "kaitlyn"),
                      "model": base, "language": lang,
                      "sample_rate": 16000, "output_format": "wav"},
            )
        if resp.status_code != 200:
            raise ProviderError(f"smallest-tts {resp.status_code}: {resp.text[:200]}")
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
    "deepgram-aura": DeepgramAuraTTS,
    "groq-tts": GroqTTS,
    "rime": RimeTTS,
    "smallest-tts": SmallestTTS,
    "mock-tts": MockTTS,
}
