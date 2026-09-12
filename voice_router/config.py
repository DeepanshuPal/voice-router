"""Load provider registry + routing defaults from YAML."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "providers.yaml"


@dataclass
class ProviderSpec:
    name: str
    capability: str  # stt | tts
    models: list[str]
    env_key: str
    languages: list[str] = field(default_factory=list)
    cost_per_minute_usd: float = 0.0
    cost_per_1k_chars_usd: float = 0.0
    default_latency_ms: float = 1000.0

    @property
    def available(self) -> bool:
        """Mock providers need no key; everything else needs its env key set."""
        return self.env_key == "" or bool(os.environ.get(self.env_key))

    def unit_cost(self, units: float) -> float:
        if self.capability == "stt":
            return self.cost_per_minute_usd * units
        return self.cost_per_1k_chars_usd * units


@dataclass
class RouterConfig:
    default_strategy: str = "benchmark"
    fallback: bool = True
    request_timeout_s: int = 30
    mock_enabled: bool = True
    stt: list[ProviderSpec] = field(default_factory=list)
    tts: list[ProviderSpec] = field(default_factory=list)

    def providers(self, capability: str) -> list[ProviderSpec]:
        """Live list for a capability ("stt" | "tts") - mutate in place."""
        return getattr(self, capability)


def load_config(path: str | os.PathLike | None = None) -> RouterConfig:
    cfg_path = Path(path or os.environ.get("ROUTER_CONFIG", DEFAULT_CONFIG_PATH))
    raw = yaml.safe_load(cfg_path.read_text())

    routing = raw.get("routing", {})
    cfg = RouterConfig(
        default_strategy=routing.get("default_strategy", "benchmark"),
        fallback=routing.get("fallback", True),
        request_timeout_s=int(routing.get("request_timeout_s", 30)),
        mock_enabled=raw.get("mock", {}).get("enabled", True),
    )

    for capability in ("stt", "tts"):
        for name, spec in (raw.get(capability) or {}).items():
            cfg.providers(capability).append(
                ProviderSpec(
                    name=name,
                    capability=capability,
                    models=spec.get("models", []),
                    env_key=spec.get("env_key", ""),
                    languages=spec.get("languages", []),
                    cost_per_minute_usd=spec.get("cost_per_minute_usd", 0.0),
                    cost_per_1k_chars_usd=spec.get("cost_per_1k_chars_usd", 0.0),
                    default_latency_ms=spec.get("default_latency_ms", 1000.0),
                )
            )

    if cfg.mock_enabled:
        cfg.stt.append(
            ProviderSpec(
                name="mock-stt",
                capability="stt",
                models=["mock-transcribe-1"],
                env_key="",
                languages=["*"],
                cost_per_minute_usd=0.0,
                default_latency_ms=50,
            )
        )
        cfg.tts.append(
            ProviderSpec(
                name="mock-tts",
                capability="tts",
                models=["mock-speak-1"],
                env_key="",
                languages=["*"],
                cost_per_1k_chars_usd=0.0,
                default_latency_ms=40,
            )
        )
    return cfg
