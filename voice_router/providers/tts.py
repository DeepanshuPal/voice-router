"""TTS adapters: ElevenLabs, OpenAI, Cartesia, Rime, Deepgram Aura, Groq Orpheus, Smallest Lightning, Hume Octave, and a zero-key mock."""

from __future__ import annotations

import os
import time

import httpx

from .base import ProviderError, ProviderUnavailable, TTSProvider, TTSTiming, sine_wav

async def _stream_audio(method_url: str, *, headers: dict, json: dict, params: dict | None = None,
                        error_name: str) -> TTSTiming:
    started = time.perf_counter()
    chunks: list[bytes] = []
    ttfb_ms = None
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", method_url, headers=headers, json=json, params=params) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")
                raise ProviderError(f"{error_name} {resp.status_code}: {body[:200]}")
            async for chunk in resp.aiter_bytes():
                if chunk:
                    if ttfb_ms is None:
                        ttfb_ms = (time.perf_counter() - started) * 1000
                    chunks.append(chunk)
    return TTSTiming(audio=b"".join(chunks), protocol="streaming_http",
                     ttfb_ms=ttfb_ms, completion_ms=(time.perf_counter() - started) * 1000)


# Default voices per provider when the caller passes an OpenAI-style name.
VOICE_MAP = {
    "elevenlabs": "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "openai-tts": "alloy",
    "cartesia": "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
}


class ElevenLabsTTS(TTSProvider):
    async def synthesize_timed(self, text: str, model: str, voice: str) -> TTSTiming:
        key = os.environ.get(self.spec.env_key)
        if not key: raise ProviderUnavailable("ELEVENLABS_API_KEY not set")
        voice_id = VOICE_MAP["elevenlabs"] if voice == "alloy" else voice
        return await _stream_audio(f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream",
            headers={"xi-api-key": key, "Content-Type": "application/json"},
            json={"text": text, "model_id": model}, error_name="elevenlabs")

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
    async def synthesize_timed(self, text: str, model: str, voice: str) -> TTSTiming:
        key = os.environ.get(self.spec.env_key)
        if not key: raise ProviderUnavailable("OPENAI_API_KEY not set")
        return await _stream_audio("https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "voice": voice, "input": text, "response_format": "wav"}, error_name="openai")

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
    async def synthesize_timed(self, text: str, model: str, voice: str) -> TTSTiming:
        key = os.environ.get(self.spec.env_key)
        if not key: raise ProviderUnavailable("CARTESIA_API_KEY not set")
        return await _stream_audio("https://api.cartesia.ai/tts/bytes",
            headers={"X-API-Key": key, "Cartesia-Version": "2024-06-10", "Content-Type": "application/json"},
            json={"model_id": model, "transcript": text,
                  "voice": {"mode": "id", "id": VOICE_MAP["cartesia"] if voice == "alloy" else voice},
                  "output_format": {"container": "wav", "encoding": "pcm_s16le", "sample_rate": 16000}}, error_name="cartesia")

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
    async def synthesize_timed(self, text: str, model: str, voice: str) -> TTSTiming:
        key = os.environ.get(self.spec.env_key)
        if not key: raise ProviderUnavailable("DEEPGRAM_API_KEY not set")
        return await _stream_audio("https://api.deepgram.com/v1/speak",
            headers={"Authorization": f"Token {key}", "Content-Type": "application/json"},
            params={"model": model}, json={"text": text}, error_name="deepgram-aura")

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


class HumeTTS(TTSProvider):
    """Hume Octave 2: per-language Hume library voices, base64 WAV in JSON.

    Octave 2 requires an explicit voice (unlike Octave 1, which generates
    one). Voices from GET /v0/tts/voices?provider=HUME_AI, filtered to
    compatible_octave_models including "2" (2026-09-14). Language rides in
    the model string ("octave-2-hi") because synthesize() gets no language.
    """

    BASE = "https://api.hume.ai/v0/tts"
    VOICE = {"en": "Colton Rivers", "es": "Spanish Instructor",
             "fr": "Mika the Musician", "de": "Anna",
             "hi": "Priya", "ja": "Akira"}

    def model_for(self, language: str) -> str:
        return f"{self.spec.models[0]}-{language}"

    async def synthesize(self, text: str, model: str, voice: str) -> bytes:
        key = os.environ.get(self.spec.env_key)
        if not key:
            raise ProviderUnavailable("HUME_API_KEY not set")
        base, _, lang = model.rpartition("-")
        payload={"utterances": [{"text": text,
                                  "voice": {"name": self.VOICE.get(lang, "Colton Rivers"),
                                            "provider": "HUME_AI"}}],
                 "version": "2", "format": {"type": "wav"}}
        async with httpx.AsyncClient(timeout=60) as client:
            for attempt in range(4):
                resp = await client.post(self.BASE,
                    headers={"X-Hume-Api-Key": key, "Content-Type": "application/json"},
                    json=payload)
                if resp.status_code != 429 or attempt == 3:
                    break
                retry_after=resp.headers.get("retry-after")
                delay=float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 2 ** attempt
                await __import__("asyncio").sleep(min(delay, 30))
        if resp.status_code != 200:
            raise ProviderError(f"hume {resp.status_code}: {resp.text[:200]}")
        import base64
        return base64.b64decode(resp.json()["generations"][0]["audio"])


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
    "hume": HumeTTS,
    "mock-tts": MockTTS,
}
