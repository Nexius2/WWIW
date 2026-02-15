from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

import requests

from ..config import TMDB_CACHE_HOURS
from ..db import meta_get

TMDB_BASE = "https://api.themoviedb.org/3"


@dataclass
class TmdbSignals:
    fetched_at: float
    trending_movie: Set[int]
    trending_tv: Set[int]
    popular_movie: Set[int]
    popular_tv: Set[int]


_cache: Optional[TmdbSignals] = None


def _api_key() -> str:
    return (meta_get("tmdb_api_key") or "").strip()


def _auth_headers() -> Dict[str, str]:
    return {"Accept": "application/json"}


def _get(path: str, params: Dict) -> Dict:
    api_key = _api_key()
    if not api_key:
        return {}
    url = f"{TMDB_BASE}{path}"
    p = dict(params or {})
    p["api_key"] = api_key
    r = requests.get(url, params=p, headers=_auth_headers(), timeout=15)
    if r.status_code != 200:
        return {}
    try:
        return r.json() or {}
    except Exception:
        return {}


def _ids_from_results(payload: Dict) -> Set[int]:
    out: Set[int] = set()
    for it in (payload.get("results") or []):
        try:
            out.add(int(it.get("id")))
        except Exception:
            pass
    return out


def fetch_signals(force: bool = False) -> Optional[TmdbSignals]:
    """Récupère (et cache) quelques signaux simples: trending & popular."""
    global _cache

    if not _api_key():
        return None

    ttl = max(1, TMDB_CACHE_HOURS) * 3600
    now = time.time()
    if (not force) and _cache and (now - _cache.fetched_at) < ttl:
        return _cache

    trending_movie = _ids_from_results(_get("/trending/movie/day", {}))
    trending_tv = _ids_from_results(_get("/trending/tv/day", {}))
    popular_movie = _ids_from_results(_get("/movie/popular", {"page": 1}))
    popular_tv = _ids_from_results(_get("/tv/popular", {"page": 1}))

    _cache = TmdbSignals(
        fetched_at=now,
        trending_movie=trending_movie,
        trending_tv=trending_tv,
        popular_movie=popular_movie,
        popular_tv=popular_tv,
    )
    return _cache


def get_sets() -> Tuple[Set[int], Set[int], Set[int], Set[int]]:
    sig = fetch_signals(force=False)
    if not sig:
        return set(), set(), set(), set()
    return sig.trending_movie, sig.trending_tv, sig.popular_movie, sig.popular_tv
