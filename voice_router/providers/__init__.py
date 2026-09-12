"""Provider adapters. Each real adapter wraps one vendor API; mock adapters
keep the whole stack runnable with zero keys."""

from .base import ProviderError, ProviderUnavailable, STTProvider, TTSProvider  # noqa: F401
