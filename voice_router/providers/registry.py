"""Build live adapter instances from loaded config."""

from __future__ import annotations

import os

from ..config import ProviderSpec, RouterConfig
from . import stt, tts
from .base import STTProvider, TTSProvider


def build_providers(cfg: RouterConfig) -> dict[str, dict[str, STTProvider | TTSProvider]]:
    out: dict[str, dict] = {"stt": {}, "tts": {}}
    raw_allowlist = os.environ.get("BENCHMARK_PROVIDER_ALLOWLIST", "")
    allowlist = {name.strip() for name in raw_allowlist.split(",") if name.strip()}
    for spec in cfg.stt:
        cls = stt.REGISTRY.get(spec.name)
        if cls and spec.available and (not allowlist or spec.name in allowlist):
            out["stt"][spec.name] = cls(spec)
    for spec in cfg.tts:
        cls = tts.REGISTRY.get(spec.name)
        if cls and spec.available and (not allowlist or spec.name in allowlist):
            out["tts"][spec.name] = cls(spec)
    return out
