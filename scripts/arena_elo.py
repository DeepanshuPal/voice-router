"""Arena ratings: Bradley-Terry Elo from raw votes, bootstrap 95% intervals.  # v2

Reads votes from the Cloudflare vote store (ARENA_EXPORT_URL + ARENA_EXPORT_TOKEN),
fits BT with ties counted as half-wins, writes docs/arena/data/ratings.json.
No third-party deps. Below MIN_VOTES it publishes a collecting state instead
of pretending rankings exist.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "arena" / "data" / "ratings.json"
MIN_VOTES = 30
BOOTSTRAP = 400


def outcome(wins: dict, v: dict) -> None:
    if v["choice"] == "a":
        wins[v["a"]] += 1
    elif v["choice"] == "b":
        wins[v["b"]] += 1
    else:
        wins[v["a"]] += 0.5
        wins[v["b"]] += 0.5


def fit_elo(votes: list[dict], players: list[str], iters: int) -> dict[str, float]:
    wins = {p: 0.0 for p in players}
    for v in votes:
        outcome(wins, v)
    w = {p: 1.0 for p in players}
    for _ in range(iters):
        for p in players:
            denom = sum(
                1.0 / (w[p] + w[v["b"] if p == v["a"] else v["a"]])
                for v in votes if p in (v["a"], v["b"])
            )
            if denom:
                w[p] = wins[p] / denom
        m = statistics.mean(w.values()) or 1.0
        w = {p: x / m for p, x in w.items()}
    return {p: 1500.0 + 400.0 * math.log10(max(w[p], 1e-12)) for p in players}


def main() -> None:
    req = urllib.request.Request(
        os.environ["ARENA_EXPORT_URL"],
        headers={"X-Export-Token": os.environ["ARENA_EXPORT_TOKEN"]},
    )
    votes = json.load(urllib.request.urlopen(req, timeout=30))["votes"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if len(votes) < MIN_VOTES:
        OUT.write_text(json.dumps({"state": "collecting", "votes": len(votes),
                                   "min_votes": MIN_VOTES, "ratings": []}, indent=2))
        print(f"only {len(votes)} votes - wrote collecting state")
        return

    players = sorted({v["a"] for v in votes} | {v["b"] for v in votes})
    games = {p: sum(1 for v in votes if p in (v["a"], v["b"])) for p in players}
    rating = fit_elo(votes, players, iters=300)

    rng = random.Random(42)
    boots: dict[str, list[float]] = {p: [] for p in players}
    for _ in range(BOOTSTRAP):
        sample = [rng.choice(votes) for _ in votes]
        r = fit_elo(sample, players, iters=100)
        for p in players:
            boots[p].append(r[p])

    rows = []
    for p in players:
        s = sorted(boots[p])
        rows.append({"provider": p, "elo": round(rating[p], 1),
                     "ci95": [round(s[int(0.025 * BOOTSTRAP)], 1),
                              round(s[int(0.975 * BOOTSTRAP)], 1)],
                     "games": games[p]})
    rows.sort(key=lambda r: -r["elo"])
    OUT.write_text(json.dumps({"state": "rated", "votes": len(votes),
                               "field_mean": 1500, "ratings": rows}, indent=2))
    print(f"rated {len(rows)} providers from {len(votes)} votes")


if __name__ == "__main__":
    main()
