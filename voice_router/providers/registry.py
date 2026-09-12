"""Build live adapter instances from loaded config."""

from __future__ import annotations

from ..config import ProviderSpec, RouterConfig
from . import stt, tts
from .base import STTProvider, TTSProvider


def build_providers(cfg: RouterConfig) -> dict[str, dict[str, STTProvider | TTSProvider]]:
    out: dict[str, dict] = {"stt": {}, "tts": {}}
    for spec in cfg.stt:
        cls = stt.REGISTRY.get(spec.name)
        if cls and spec.available:
            out["stt"][spec.name] = cls(spec)
    for spec in cfg.tts:
        cls = tts.REGISTRY.get(spec.name)
        if cls and spec.available:
            out["tts"][spec.name] = cls(spec)
    return out
