from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional, Tuple

def _safe_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _duration_min(duration_ms) -> Optional[int]:
    try:
        if duration_ms:
            return int(round(float(duration_ms) / 60000.0))
    except Exception:
        pass
    return None

def _tags(tag_objs) -> List[str]:
    out: List[str] = []
    for t in (tag_objs or []):
        tag = getattr(t, "tag", None)
        if tag:
            out.append(tag)
    return out

_GENRE_ALIASES = {
    "anime": "animation",
    "animated": "animation",
    "animation": "animation",
}

def _norm_genre(g: str) -> str:
    x = (g or "").strip().casefold()
    return _GENRE_ALIASES.get(x, x)

def _to_stars(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if v <= 0:
        return None
    if v > 10.0:          # % (0..100)
        return max(0.0, min(5.0, v / 20.0))
    return max(0.0, min(5.0, v / 2.0))  # /10 -> /5

def _to_stars_5(value_10: Optional[float]) -> Optional[float]:
    # Plex rating est souvent sur 10. Convertit en /5.
    if value_10 is None:
        return None
    try:
        v = float(value_10)
    except Exception:
        return None
    if v <= 0:
        return None
    return max(0.0, min(5.0, v / 2.0))

def _stars_str(stars_5: Optional[float]) -> Optional[str]:
    if stars_5 is None:
        return None
    s = round(stars_5 * 2) / 2.0
    full = int(s)
    half = 1 if (s - full) >= 0.5 else 0
    empty = max(0, 5 - full - half)
    return ("★" * full) + ("½" if half else "") + ("☆" * empty)

def format_ts(ts: Optional[str]) -> Optional[str]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z",""))
        return dt.strftime("%d/%m/%Y %H:%M")
    except Exception:
        return ts

def form_int(v: Optional[str]) -> Optional[int]:
    if v is None:
        return None
    v = str(v).strip()
    if v == "":
        return None
    try:
        return int(v)
    except Exception:
        return None

def form_float(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    v = str(v).strip()
    if v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None

def extract_provider_ratings(media) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    imdb = tmdb = tvdb = None

    for r in (getattr(media, "ratings", None) or []):
        img = str(getattr(r, "image", "") or "").lower()
        val = _safe_float(getattr(r, "value", None))
        if val is None:
            continue
        if "imdb" in img:
            imdb = val
        elif "themoviedb" in img or "tmdb" in img:
            tmdb = val
        elif "thetvdb" in img or "tvdb" in img:
            tvdb = val

    if imdb is None or tmdb is None or tvdb is None:
        for img_attr, val_attr in (("ratingImage", "rating"), ("audienceRatingImage", "audienceRating")):
            img = str(getattr(media, img_attr, "") or "").lower()
            val = _safe_float(getattr(media, val_attr, None))
            if val is None:
                continue
            if imdb is None and "imdb" in img:
                imdb = val
            elif tmdb is None and ("themoviedb" in img or "tmdb" in img):
                tmdb = val
            elif tvdb is None and ("thetvdb" in img or "tvdb" in img):
                tvdb = val

    return imdb, tmdb, tvdb
