"""Daily recommendation: context + style centroid query, scored with anti-repeat decay.

    E_weather+vibe = Norm( w * E(vibe phrase) + (1 - w) * E(weather + day phrase) )
    Q_day    = Norm( alpha * E_weather+vibe + (1 - alpha) * U_style_centroid )
    Score(i) = (Q_day . V_i) * (1 - exp(-lambda * dt_i))       dt_i = days since last worn

The vibe and weather phrases are embedded separately and mixed: embedding them as one sentence
lets strong weather words ("rainy") swamp the vibe pill (Gate 3 finding).

On top of the spec formula, two transparent multipliers (both reported per result):
  * context fit  - +VIBE_BOOST / +SEASON_BOOST when the outfit's vibe/season tags match today, and a
                   0.7x weather guard for fabrics that fail in rain (silk, velvet...) or heat (wool...)
  * venue radar  - VENUE_REPEAT_PENALTY when the look is among the last VENUE_RECENT_LOOKS distinct
                   looks worn at today's venue within VENUE_REPEAT_DAYS (the "same group, same look"
                   warning). Counting looks rather than days keeps daily venues (office) from
                   flagging the whole work wardrobe while still covering rare venues for months.
"""
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from engine import config, db
from engine.context import season_from_weather, venue_labels
from engine.taxonomy import HOT, VIBES, WET

VIBE_PHRASES = {
    "Work": "smart professional office workwear",
    "Casual": "relaxed casual everyday outfit",
    "Festive": "festive traditional celebration outfit for a function",
    "Travel": "comfortable practical travel outfit",
}
RAIN_UNSAFE = {"silk", "raw silk", "velvet", "brocade", "organza", "leather", "tweed", "wool"}
HEAT_UNSAFE = {"wool", "tweed", "velvet", "leather", "brocade"}


@dataclass
class DayContext:
    vibe: str = "Casual"
    when: datetime = field(default_factory=datetime.now)
    temp_c: float | None = None
    humidity: float | None = None
    rain: bool = False
    venue_cluster: str | None = None   # where today happens (e.g. office cluster) for the radar

    def __post_init__(self):
        if self.vibe not in VIBES:
            raise ValueError(f"vibe must be one of {VIBES}")

    @property
    def season(self) -> str | None:
        return season_from_weather(self.temp_c, self.humidity, self.rain)

    def phrases(self) -> tuple[str, str]:
        """(vibe phrase, weather + day phrase)."""
        weather = []
        if self.temp_c is not None:
            feel = ("very hot" if self.temp_c >= 33 else "hot" if self.temp_c >= 28 else
                    "warm" if self.temp_c >= 23 else "mild" if self.temp_c >= 16 else "cold")
            weather.append(f"{feel} {self.temp_c:.0f} degree")
        if self.humidity is not None and self.humidity >= 70:
            weather.append("humid")
        if self.rain:
            weather.append("rainy monsoon")
        h = self.when.hour
        tod = "morning" if h < 12 else "afternoon" if h < 17 else "evening"
        day = f"outfit for {' '.join(weather)} weather" if weather else "outfit"
        return VIBE_PHRASES[self.vibe], f"{day} on a {self.when.strftime('%A')} {tod}"

    def describe(self) -> str:
        return " | ".join(self.phrases())


def style_centroid(ids: list[int], vecs: np.ndarray, wear_counts: dict[int, int]) -> np.ndarray:
    """Wear-weighted mean of the wardrobe: 'what this person actually likes wearing'."""
    w = np.array([math.log1p(wear_counts.get(i, 1)) for i in ids], dtype=np.float32)
    c = (vecs * w[:, None]).sum(axis=0)
    return c / (np.linalg.norm(c) or 1.0)


def recency_factor(dt_days: float | None, lam: float = config.DECAY_LAMBDA) -> float:
    if dt_days is None:
        return 1.0
    return 1.0 - math.exp(-lam * max(dt_days, 0.0))


class Recommender:
    def __init__(self, conn: sqlite3.Connection, embedder=None, alpha: float = config.ALPHA,
                 lam: float = config.DECAY_LAMBDA):
        from engine.embedder import get_embedder

        self.conn = conn
        self.embedder = embedder or get_embedder()
        self.alpha = alpha
        self.lam = lam
        self._ctx_cache: dict[str, np.ndarray] = {}

    def context_vector(self, ctx: DayContext) -> np.ndarray:
        key = ctx.describe()
        if key not in self._ctx_cache:
            e_vibe, e_day = self.embedder.embed_texts(list(ctx.phrases()), task="search")
            v = config.VIBE_WEIGHT * e_vibe + (1 - config.VIBE_WEIGHT) * e_day
            self._ctx_cache[key] = v / np.linalg.norm(v)
        return self._ctx_cache[key]

    def recommend(self, ctx: DayContext, top_k: int = 3) -> dict:
        ids, vecs = db.all_embeddings(self.conn)
        if not ids:
            return {"context": ctx.describe(), "results": []}
        outfits = {r["id"]: db.outfit_dict(r) for r in self.conn.execute("SELECT * FROM outfits")}
        when = ctx.when.isoformat()
        # Wear history strictly before 'when' (lets the CLI backtest replay the past).
        history: dict[int, list[sqlite3.Row]] = {}
        for r in self.conn.execute(
                "SELECT outfit_id, worn_at, location_cluster FROM wear_logs WHERE worn_at < ? ORDER BY worn_at DESC",
                (when,)):
            history.setdefault(r["outfit_id"], []).append(r)
        venues = venue_labels(self.conn)
        recent_at_venue: set[int] = set()
        if ctx.venue_cluster and venues.get(ctx.venue_cluster) != "Home":  # nobody judges repeats at home
            since = (ctx.when - timedelta(days=config.VENUE_REPEAT_DAYS)).isoformat()
            for r in self.conn.execute(
                    "SELECT outfit_id FROM wear_logs WHERE location_cluster = ? AND worn_at >= ? AND worn_at < ?"
                    " GROUP BY outfit_id ORDER BY MAX(worn_at) DESC LIMIT ?",
                    (ctx.venue_cluster, since, when, config.VENUE_RECENT_LOOKS)):
                recent_at_venue.add(r["outfit_id"])

        e_ctx = self.context_vector(ctx)
        u = style_centroid(ids, vecs, {i: len(history.get(i, [])) or 1 for i in ids})
        q = self.alpha * e_ctx + (1 - self.alpha) * u
        q /= np.linalg.norm(q)
        sims = vecs @ q
        season = ctx.season

        results = []
        for i, oid in enumerate(ids):
            o = outfits[oid]
            past = history.get(oid, [])
            dt = (ctx.when - datetime.fromisoformat(past[0]["worn_at"])).total_seconds() / 86400 if past else None
            decay = recency_factor(dt, self.lam)
            base = float(sims[i]) * decay

            fit, reasons, warnings = 1.0, [], []
            if o["vibe_tag"] == ctx.vibe:
                fit += config.VIBE_BOOST
                reasons.append(f"{ctx.vibe} look")
            if season and o["season_tag"] == season:
                fit += config.SEASON_BOOST
                reasons.append(f"suits {season}")
            fabric = (o["fabric_tag"] or "").lower()
            if ctx.rain and fabric in RAIN_UNSAFE:
                fit *= 0.7
                warnings.append(f"{fabric} is risky in rain")
            if season == HOT and fabric in HEAT_UNSAFE:
                fit *= 0.7
                warnings.append(f"{fabric} runs hot")

            radar = 1.0
            if dt is not None and dt < 7:
                days = (ctx.when.date() - datetime.fromisoformat(past[0]["worn_at"]).date()).days
                warnings.append("worn today" if days == 0 else "worn yesterday" if days == 1 else f"worn {days} days ago")
            if oid in recent_at_venue:
                hit = next(p for p in past if p["location_cluster"] == ctx.venue_cluster)
                radar = config.VENUE_REPEAT_PENALTY
                days = (ctx.when - datetime.fromisoformat(hit["worn_at"])).days
                warnings.append(f"same look at {venues.get(ctx.venue_cluster, ctx.venue_cluster)} {days}d ago")

            results.append({
                "outfit_id": oid,
                "primary_category": o["primary_category"],
                "vibe_tag": o["vibe_tag"], "season_tag": o["season_tag"], "fabric_tag": o["fabric_tag"],
                "crop_image_path": o["crop_image_path"],
                "wear_count": o["wear_count"],
                "days_since_worn": None if dt is None else round(dt, 1),
                # multipliers must push negative similarities further down, not up
                "score": base * fit * radar if base >= 0 else base / (fit * radar),
                "breakdown": {"similarity": round(float(sims[i]), 4), "recency": round(decay, 4),
                              "context_fit": round(fit, 3), "venue_radar": radar},
                "reasons": reasons, "warnings": warnings,
            })
        results.sort(key=lambda r: r["score"], reverse=True)
        return {"context": ctx.describe(), "season": season, "results": results[:top_k]}
