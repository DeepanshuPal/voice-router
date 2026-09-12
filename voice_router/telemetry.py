"""Per-call telemetry: append-only JSONL plus in-memory aggregates."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path

from .models import CallRecord


class Telemetry:
    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path or os.environ.get("TELEMETRY_PATH", ".telemetry/calls.jsonl"))
        self._records: list[CallRecord] = []
        self._lock = threading.Lock()

    def record(self, rec: CallRecord) -> None:
        with self._lock:
            self._records.append(rec)
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as f:
                    f.write(rec.model_dump_json() + "\n")
            except OSError:
                pass  # telemetry must never break a call

    def p50_latency(self, capability: str, provider: str) -> float | None:
        with self._lock:
            vals = sorted(
                r.latency_ms
                for r in self._records
                if r.capability == capability and r.provider == provider and r.status == "ok"
            )
        if not vals:
            return None
        return vals[len(vals) // 2]

    def summary(self) -> dict:
        with self._lock:
            recs = list(self._records)
        by_provider: dict[str, dict] = defaultdict(
            lambda: {"calls": 0, "errors": 0, "cost_usd": 0.0, "latencies": []}
        )
        for r in recs:
            b = by_provider[f"{r.capability}:{r.provider}"]
            b["calls"] += 1
            b["errors"] += r.status != "ok"
            b["cost_usd"] += r.cost_usd
            if r.status == "ok":
                b["latencies"].append(r.latency_ms)
        out = {}
        for key, b in by_provider.items():
            lat = sorted(b.pop("latencies"))
            out[key] = {
                **b,
                "cost_usd": round(b["cost_usd"], 6),
                "p50_latency_ms": lat[len(lat) // 2] if lat else None,
            }
        return {"total_calls": len(recs), "providers": out}

    @staticmethod
    def now() -> float:
        return time.time()
