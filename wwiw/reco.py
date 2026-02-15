from __future__ import annotations

import json
import random
import threading
from urllib.parse import quote
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import DEFAULT_LIMIT
from .db import q_all, exec_sql, meta_get
from .state import reco_recent_keys, reco_remember

from .utils import _norm_genre, _stars_str, _to_stars, _to_stars_5

try:
    from .external.tmdb import get_sets as tmdb_get_sets
except Exception:
    tmdb_get_sets = None


@dataclass
class RecItem:
    server_id: int
    server: str
    library: str
    base_url: Optional[str]
    machine_id: Optional[str]
    rating_key: str
    type_raw: str             # movie/show
    type: str                 # Film/Série (UI)
    title: str
    year: Optional[int]
    duration_min: Optional[int]
    rating: Optional[float]
    audience_rating: Optional[float]
    imdb_id: Optional[str] = None
    tmdb_id: Optional[int] = None
    tvdb_id: Optional[int] = None
    rating_imdb: Optional[float] = None
    rating_tmdb: Optional[float] = None
    rating_tvdb: Optional[float] = None
    watched: bool = False
    genres: List[str] = None
    summary: str = ""
    thumb: Optional[str] = None


def top_genres_profile() -> List[str]:
    rows = q_all("SELECT watched, genres_json FROM items")
    if not rows:
        return []
    watched = [r for r in rows if int(r["watched"]) == 1]
    source = watched if watched else rows
    c = Counter()
    for r in source:
        try:
            gs = json.loads(r["genres_json"] or "[]")
        except Exception:
            gs = []
        for g in gs:
            c[g] += 1
    return [g for g, _ in c.most_common(5)]


_model_lock = threading.Lock()
_model_cache = {
    "built_for_scan_ts": None,
    "vectorizer": None,
    "matrix": None,
    "row_keys": None,   # List[(server_id, rating_key)] aligné avec matrix rows
}

# Cache machine_id per Plex server to build Plex Web deep-links even before a full sync
_server_mid_cache: Dict[int, str] = {}
_server_mid_lock = threading.Lock()


def _get_server_machine_id(server_id: int, base_url: str, token: str) -> Optional[str]:
    """Return Plex machineIdentifier for a server.

    - Uses an in-process cache.
    - Falls back to a lightweight Plex API call.
    - Persists the machine_id in DB when found.
    """
    if not server_id or not base_url or not token:
        return None

    with _server_mid_lock:
        mid = _server_mid_cache.get(server_id)
    if mid:
        return mid

    try:
        # Plex API call is cheap and avoids forcing a full sync just to get machine_id
        from .plex import connect_plex

        plex = connect_plex(base_url, token)
        mid = str(getattr(plex, "machineIdentifier", "") or "").strip()
        if not mid:
            return None

        with _server_mid_lock:
            _server_mid_cache[server_id] = mid

        try:
            exec_sql("UPDATE servers SET machine_id=? WHERE id=?", (mid, server_id))
        except Exception:
            pass

        return mid
    except Exception:
        return None


def _get_last_scan_ts() -> Optional[str]:
    return meta_get("last_scan_ts")

def _doc_for_item(title: str, genres: List[str], summary: str, year: Optional[int]) -> str:
    g = " ".join([_norm_genre(x) for x in (genres or [])])
    y = str(year) if year else ""
    s = (summary or "")
    return f"{title} {title} {g} {g} {y} {s}"

def _ensure_model() -> None:
    scan_ts = _get_last_scan_ts()

    with _model_lock:
        if _model_cache["vectorizer"] is not None and _model_cache["built_for_scan_ts"] == scan_ts:
            return

        rows = q_all(
            """
            SELECT server_id, rating_key, title, year, genres_json, summary
            FROM items
            ORDER BY updated_at DESC
            LIMIT 12000
            """
        )

        docs: List[str] = []
        row_keys: List[Tuple[int, str]] = []

        for r in rows:
            try:
                sid = int(r["server_id"])
                rk = str(r["rating_key"])
            except Exception:
                continue
            try:
                gs = json.loads(r["genres_json"] or "[]")
            except Exception:
                gs = []
            docs.append(_doc_for_item(str(r["title"] or ""), gs, str(r["summary"] or ""), r["year"]))
            row_keys.append((sid, rk))

        if not docs:
            _model_cache.update({"built_for_scan_ts": scan_ts, "vectorizer": None, "matrix": None, "row_keys": None})
            return

        vec = TfidfVectorizer(
            stop_words=None,
            strip_accents="unicode",
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
        )
        mat = vec.fit_transform(docs)

        _model_cache["built_for_scan_ts"] = scan_ts
        _model_cache["vectorizer"] = vec
        _model_cache["matrix"] = mat
        _model_cache["row_keys"] = row_keys

def _profile_vectors(type_filter: Optional[str] = None) -> tuple[Optional[Any], Optional[Any]]:
    """
    Construit deux profils:
    - profil POSITIF : choses aimées (terminées, rewatch, bien notées)
    - profil NEGATIF : choses rejetées (abandonnées tôt, mal notées, pas rewatch)

    Important:
    - On ne limite pas aux N derniers: on pondère par récence (demi-vie)
    - Un contenu rewatch (view_count>=2) n'alimente JAMAIS le profil négatif,
      même si la dernière lecture a été interrompue rapidement.
    """
    with _model_lock:
        mat = _model_cache["matrix"]
        row_keys = _model_cache["row_keys"] or []
        if mat is None or not row_keys:
            return None, None

    # Réglages (meta, avec defaults)
    half_life_days = float(meta_get("reco_half_life_days") or 180.0)
    rewatch_boost = float(meta_get("reco_rewatch_boost") or 0.20)  # +20% si rewatch
    missing_last_viewed_weight = float(meta_get("reco_missing_last_viewed_weight") or 0.25)

    # seuils "préférence"
    pos_completion = float(meta_get("reco_pos_completion") or 0.85)
    neg_completion = float(meta_get("reco_neg_completion") or 0.15)
    pos_rating = float(meta_get("reco_pos_user_rating") or 8.0)
    neg_rating = float(meta_get("reco_neg_user_rating") or 4.0)

    min_progress_ms = int(float(meta_get("reco_min_progress_ms") or 300000))  # 5 min

    sql = """
        SELECT server_id, rating_key, last_viewed_at, view_count,
               watched, user_rating, completion_ratio, view_offset_ms, duration_ms, type
        FROM items
        WHERE (watched=1 OR (view_offset_ms IS NOT NULL AND view_offset_ms > 0))
    """
    params: tuple = ()
    if type_filter in ("movie", "show"):
        sql += " AND type=?"
        params = (type_filter,)

    rows = q_all(sql, params)
    if not rows:
        return None, None

    with _model_lock:
        key_to_idx = {k: i for i, k in enumerate(_model_cache["row_keys"] or [])}

        now = datetime.utcnow().timestamp()

        pos_idx: list[int] = []
        pos_w: list[float] = []
        neg_idx: list[int] = []
        neg_w: list[float] = []

        # poids récence: 2^(-age_days/half_life_days)
        def recency_weight(lv: Any) -> float:
            if lv:
                try:
                    age_days = max(0.0, (now - float(lv)) / 86400.0)
                    return 2.0 ** (-age_days / max(1e-6, half_life_days))
                except Exception:
                    return missing_last_viewed_weight
            return missing_last_viewed_weight

        for r in rows:
            k = (int(r["server_id"]), str(r["rating_key"]))
            idx = key_to_idx.get(k)
            if idx is None:
                continue

            lv = r["last_viewed_at"]
            vc = r["view_count"]
            watched_flag = r["watched"]
            ur = r["user_rating"]
            cr = r["completion_ratio"]
            vo = r["view_offset_ms"]
            dm = r["duration_ms"]

            try:
                vc_int = int(vc) if vc is not None else 0
            except Exception:
                vc_int = 0

            w = recency_weight(lv)

            # rewatch => fort signal positif + jamais négatif
            if vc_int >= 2:
                w_pos = w * (1.0 + rewatch_boost)
                pos_idx.append(idx)
                pos_w.append(w_pos)
                continue

            # signaux user rating / completion / progress
            ur_f = None
            try:
                if ur is not None:
                    ur_f = float(ur)
            except Exception:
                ur_f = None

            cr_f = None
            try:
                if cr is not None:
                    cr_f = float(cr)
            except Exception:
                cr_f = None

            # progress brut en ms (pour distinguer "lancé 30s" vs "vrai arrêt à 10min")
            vo_i = None
            try:
                if vo is not None:
                    vo_i = int(vo)
            except Exception:
                vo_i = None

            watched_i = 0
            try:
                watched_i = int(watched_flag) if watched_flag is not None else 0
            except Exception:
                watched_i = 0

            # décide POS / NEG
            is_pos = False
            is_neg = False

            # 1) user rating => très fort signal, et on module le poids selon la note
            if ur_f is not None:
                if ur_f >= pos_rating:
                    is_pos = True
                    # note haute => plus de poids (8->1.05, 10->1.25)
                    w *= (1.0 + min(0.25, max(0.0, (ur_f - pos_rating) / 10.0)))
                elif ur_f <= neg_rating:
                    is_neg = True
                    # note basse => négatif un peu plus fort (0->1.25)
                    w *= (1.0 + min(0.25, max(0.0, (neg_rating - ur_f) / 10.0)))

            # 2) completion ratio => bon signal
            if cr_f is not None:
                if cr_f >= pos_completion:
                    is_pos = True
                elif cr_f <= neg_completion and not is_pos:
                    # seulement négatif si ce n'est pas déjà positif
                    is_neg = True

            # 3) Si pas "watched" et pas de completion, on peut quand même détecter un vrai abandon
            #    avec view_offset_ms: on ignore les micro-démarrages (<5 min)
            if watched_i == 0 and cr_f is None and vo_i is not None:
                if vo_i >= min_progress_ms and not is_pos:
                    is_neg = True

            # fallback: si c'est marqué watched, mais pas de signaux -> petit positif
            if not is_pos and not is_neg and watched_i == 1:
                is_pos = True
                w *= 0.35

            # si c'est juste un micro-start non watched => on ignore
            if watched_i == 0 and not is_pos and not is_neg:
                continue

            # fallback: si marqué watched mais aucun signal exploitable => petit positif
            if watched_i == 1 and not is_pos and not is_neg:
                is_pos = True
                w *= 0.35


            if is_pos:
                pos_idx.append(idx)
                pos_w.append(w)
            if is_neg:
                # négatif plus modéré par défaut (sinon ça sur-filtre)
                neg_idx.append(idx)
                neg_w.append(w * 0.75)

        def weighted_mean(indices: list[int], weights: list[float]) -> Optional[Any]:
            if not indices:
                return None
            W = np.asarray(weights, dtype=float)
            if W.sum() <= 0:
                return None
            prof = (mat[indices].multiply(W[:, None])).sum(axis=0) / W.sum()
            return np.asarray(prof)

        return weighted_mean(pos_idx, pos_w), weighted_mean(neg_idx, neg_w)

def _rating_best(it: RecItem) -> Optional[float]:
    for v in (it.rating, it.audience_rating, it.rating_imdb, it.rating_tmdb, it.rating_tvdb):
        if v is None:
            continue
        try:
            fv = float(v)
            if fv > 0:
                return fv
        except Exception:
            pass
    return None

def _normalize_rating_0_1(v10: Optional[float]) -> float:
    if v10 is None:
        return 0.0
    try:
        x = float(v10)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, x / 10.0))

def _age_bonus(year: Optional[int]) -> float:
    if not year:
        return 0.0
    now_year = datetime.utcnow().year
    age = max(0, now_year - int(year))
    return max(0.0, 0.15 - (age * 0.01))

def _tmdb_boost(it: RecItem) -> float:
    # Clé TMDB stockée en DB (UI Servers)
    tmdb_key = (meta_get("tmdb_api_key") or "").strip()
    if not tmdb_key or not tmdb_get_sets or not it.tmdb_id:
        return 0.0

    # tmdb_id peut arriver en str via sqlite selon le row_factory / data
    try:
        tmdb_id = int(it.tmdb_id)
    except Exception:
        return 0.0

    try:
        trending_movie, trending_tv, popular_movie, popular_tv = tmdb_get_sets()
    except Exception:
        return 0.0

    if it.type_raw == "movie":
        if tmdb_id in trending_movie:
            return 0.25
        if tmdb_id in popular_movie:
            return 0.12

    if it.type_raw == "show":
        if tmdb_id in trending_tv:
            return 0.25
        if tmdb_id in popular_tv:
            return 0.12

    return 0.0


def score_item_smart(it: RecItem, top_genres: List[str], sim_score: float) -> float:
    top_set = {_norm_genre(x) for x in (top_genres or [])}
    s = float(sim_score) * 6.0
    for g in (it.genres or []):
        if _norm_genre(g) in top_set:
            s += 1.2
    s += _normalize_rating_0_1(_rating_best(it)) * 3.5
    s += _age_bonus(it.year)
    s += _tmdb_boost(it)
    if not it.genres:
        s -= 0.3
    if not (it.summary or "").strip():
        s -= 0.2
    return s

def diversify(scored: List[Tuple[float, RecItem]], limit: int) -> List[Tuple[float, RecItem]]:
    picked: List[Tuple[float, RecItem]] = []
    cnt = Counter()
    max_per = max(2, int(limit * 0.4))

    def main_genre(x: RecItem) -> str:
        return _norm_genre(x.genres[0]) if x.genres else "unknown"

    for s, it in scored:
        if len(picked) >= limit:
            break
        g = main_genre(it)
        if cnt[g] >= max_per:
            continue
        cnt[g] += 1
        picked.append((s, it))

    if len(picked) < limit:
        already = {(it.server, it.library, it.title, it.year) for _, it in picked}
        for s, it in scored:
            if len(picked) >= limit:
                break
            key = (it.server, it.library, it.title, it.year)
            if key in already:
                continue
            picked.append((s, it))

    return picked

def recommend(filters: Dict[str, Any]) -> Dict[str, Any]:
    limit = int(filters.get("limit") or DEFAULT_LIMIT)
    wanted_type = filters.get("type") or "both"
    include_watched = bool(filters.get("include_watched"))
    max_age_years = filters.get("max_age_years")
    min_rating = filters.get("min_rating")
    min_stars = filters.get("min_stars")

    # TMDB key stockée en DB (UI)
    tmdb_key = (meta_get("tmdb_api_key") or "").strip()

    min_rating_from_stars = None
    if isinstance(min_stars, (int, float)):
        min_rating_from_stars = float(min_stars) * 2.0

    duration_min = filters.get("duration_min")
    duration_max = filters.get("duration_max")
    servers = set(filters.get("servers") or [])
    libraries_ids = set()
    for x in (filters.get("libraries_ids") or []):
        try:
            libraries_ids.add(int(str(x).strip()))
        except Exception:
            pass

    include_genres = {_norm_genre(g) for g in (filters.get("include_genres") or [])}
    exclude_genres = {_norm_genre(g) for g in (filters.get("exclude_genres") or [])}
    
    # Anti-répétition (cooldown)
    # - si tu recliques sur "Générer", on exclut temporairement les items déjà proposés
    # - valeur par défaut: 120s
    # - configurable via meta: reco_cooldown_sec
    try:
        cooldown_sec = int(float((meta_get("reco_cooldown_sec") or "120")))
    except Exception:
        cooldown_sec = 120
    cooldown_sec = max(0, cooldown_sec)
    avoid_keys = reco_recent_keys() if cooldown_sec > 0 else set()


    now_year = datetime.utcnow().year
    min_year = None
    if isinstance(max_age_years, int) and max_age_years > 0:
        min_year = now_year - max_age_years

    wh = []
    params: List[Any] = []

    if wanted_type in ("movie", "show"):
        wh.append("i.type=?")
        params.append(wanted_type)

    if not include_watched:
        wh.append("i.watched=0")

    if min_year is not None:
        wh.append("(i.year IS NULL OR i.year>=?)")
        params.append(min_year)

    if isinstance(duration_min, int) and duration_min > 0:
        if wanted_type == "movie":
            wh.append("(i.duration_min IS NULL OR i.duration_min>=?)")
            params.append(duration_min)
        elif wanted_type == "both":
            wh.append("(i.type!='movie' OR i.duration_min IS NULL OR i.duration_min>=?)")
            params.append(duration_min)

    if isinstance(duration_max, int) and duration_max > 0:
        if wanted_type == "movie":
            wh.append("(i.duration_min IS NULL OR i.duration_min<=?)")
            params.append(duration_max)
        elif wanted_type == "both":
            wh.append("(i.type!='movie' OR i.duration_min IS NULL OR i.duration_min<=?)")
            params.append(duration_max)


    effective_min_rating = None
    if isinstance(min_rating, (int, float)):
        effective_min_rating = float(min_rating)
    elif isinstance(min_rating_from_stars, (int, float)):
        effective_min_rating = float(min_rating_from_stars)

    if isinstance(effective_min_rating, (int, float)):
        wh.append(
            "((i.rating IS NOT NULL AND i.rating>=?) OR "
            "(i.rating IS NULL AND i.audience_rating IS NOT NULL AND i.audience_rating>=?))"
        )
        params.extend([float(effective_min_rating), float(effective_min_rating)])

    sql = """
    SELECT
      s.id as server_id,
      s.name as server_name,
      s.base_url as server_base_url,
      s.machine_id as server_machine_id,
      s.token as server_token,
      l.title as library_title,
      i.library_id,
      i.rating_key,
      i.type,
      i.title, i.year, i.duration_min,
      i.rating, i.audience_rating,
      i.watched, i.genres_json, i.summary,
      i.imdb_id, i.tmdb_id, i.tvdb_id,
      i.rating_imdb, i.rating_tmdb, i.rating_tvdb,
      i.thumb
    FROM items i
    JOIN servers s ON s.id=i.server_id
    JOIN libraries l ON l.id=i.library_id
    WHERE s.enabled=1 AND l.enabled=1
    """

    # ✅ Filtrer les bibliothèques directement en SQL (sinon LIMIT 4000 casse tout)
    if libraries_ids:
        lids = sorted(libraries_ids)
        placeholders = ",".join(["?"] * len(lids))
        wh.append(f"i.library_id IN ({placeholders})")
        params.extend(lids)

    if wh:
        sql += " AND " + " AND ".join(wh)

    # tu peux laisser 4000, mais je te conseille plus large (surtout si multi-serveurs)
    sql += " ORDER BY i.updated_at DESC LIMIT 12000"


    rows = q_all(sql, tuple(params))
    if not rows:
        return {"ok": False, "error": "Aucun item en DB. Fais un scan."}

    _ensure_model()
    # # Profils séparés Film / Série, en version POSITIF / NEGATIF
    pos_movie, neg_movie = _profile_vectors("movie")
    pos_show, neg_show = _profile_vectors("show")

    # Poids du "profil négatif" (éviter ce que tu as tendance à abandonner / mal noter)
    neg_alpha = float(meta_get("reco_neg_alpha") or 0.35)

    with _model_lock:
        mat = _model_cache.get("matrix")
        row_keys = _model_cache.get("row_keys") or []

    sims_movie_by_key: Dict[Tuple[int, str], float] = {}
    sims_show_by_key: Dict[Tuple[int, str], float] = {}

    def _fill_sims(pos_prof, neg_prof, out: Dict[Tuple[int, str], float]) -> None:
        if mat is None or not row_keys:
            return
        base = None
        if pos_prof is not None:
            pv = np.asarray(pos_prof)
            if pv.ndim == 1:
                pv = pv.reshape(1, -1)
            base = cosine_similarity(mat, pv).reshape(-1)
        else:
            base = np.zeros((mat.shape[0],), dtype=float)

        if neg_prof is not None:
            nv = np.asarray(neg_prof)
            if nv.ndim == 1:
                nv = nv.reshape(1, -1)
            neg = cosine_similarity(mat, nv).reshape(-1)
            base = base - (neg_alpha * neg)

        for i, k in enumerate(row_keys):
            try:
                out[k] = float(base[i])
            except Exception:
                pass

    _fill_sims(pos_movie, neg_movie, sims_movie_by_key)
    _fill_sims(pos_show, neg_show, sims_show_by_key)

    top_genres = top_genres_profile()

    server_tokens: Dict[int, str] = {}

    scored: List[Tuple[float, RecItem]] = []
    for r in rows:
        server_id = int(r["server_id"])
        server_name = str(r["server_name"])
        library_title = str(r["library_title"])
        rating_key = str(r["rating_key"])

        # token (for machine_id lookup if needed)
        try:
            server_tokens[server_id] = str(r["server_token"] or "")
        except Exception:
            server_tokens[server_id] = server_tokens.get(server_id, "")

        # évite les propositions récentes (cooldown)
        if avoid_keys and (server_id, rating_key) in avoid_keys:
            continue


        if servers and server_name not in servers:
            continue
        if libraries_ids and int(r["library_id"]) not in libraries_ids:
            continue


        try:
            gs = json.loads(r["genres_json"] or "[]")
        except Exception:
            gs = []
        gs_norm = {_norm_genre(x) for x in gs}

        if include_genres and not (gs_norm & include_genres):
            continue
        if exclude_genres and (gs_norm & exclude_genres):
            continue

        # tmdb_id peut sortir en str/None -> cast safe
        tmdb_id_val = r["tmdb_id"]
        try:
            tmdb_id_val = int(tmdb_id_val) if tmdb_id_val is not None else None
        except Exception:
            tmdb_id_val = None

        it = RecItem(
            server_id=server_id,
            server=server_name,
            library=library_title,
            base_url=(r["server_base_url"] if "server_base_url" in r.keys() else None),
            machine_id=(r["server_machine_id"] if "server_machine_id" in r.keys() else None),
            rating_key=rating_key,
            type_raw=str(r["type"]),
            type=str(r["type"]),
            title=str(r["title"]),
            year=r["year"],
            duration_min=r["duration_min"],
            rating=r["rating"],
            audience_rating=r["audience_rating"],
            imdb_id=r["imdb_id"],
            tmdb_id=tmdb_id_val,
            tvdb_id=r["tvdb_id"],
            rating_imdb=r["rating_imdb"],
            rating_tmdb=r["rating_tmdb"],
            rating_tvdb=r["rating_tvdb"],
            watched=bool(int(r["watched"])),
            genres=gs,
            summary=(str(r["summary"] or "")[:240] + ("…" if r["summary"] and len(str(r["summary"])) > 240 else "")),
            thumb=r["thumb"],
        )

        if it.type_raw == "show":
            sim = float(sims_show_by_key.get((server_id, rating_key), 0.0))
        else:
            sim = float(sims_movie_by_key.get((server_id, rating_key), 0.0))
        # garder la similarité brute pour l'UI (explications)
        try:
            it.sim = sim
        except Exception:
            pass
        scored.append((score_item_smart(it, top_genres, sim), it))

    if not scored:
        return {"ok": False, "error": "Aucun résultat avec ces filtres."}

    scored.sort(key=lambda x: x[0], reverse=True)
    picked = diversify(scored, limit)

    # Exploration (optionnel) : remplace une partie des picks par des candidats "bons mais différents"
    explore_pct = float(meta_get("reco_explore_pct") or 0.15)  # 0.0 => off
    if explore_pct > 0 and limit > 0:
        try:
            explore_n = int(round(limit * explore_pct))
        except Exception:
            explore_n = 0

        if explore_n > 0 and len(scored) > limit:
            top_k = int(meta_get("reco_explore_top_k") or 250)
            top_k = max(limit * 3, min(top_k, len(scored)))
            pool = scored[:top_k]

            picked_keys = {(it.server_id, it.rating_key) for _, it in picked}
            candidates = [(s, it) for s, it in pool if (it.server_id, it.rating_key) not in picked_keys]

            if candidates:
                # tirage pondéré par le score (clamp) pour éviter de tirer n'importe quoi
                weights = [max(0.0001, float(s) + 1.0) for s, _ in candidates]
                chosen = []
                for _ in range(min(explore_n, len(candidates))):
                    idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
                    chosen.append(candidates.pop(idx))
                    weights.pop(idx)

                # Remplacer les derniers items (les moins forts) par ces "explorations"
                if chosen:
                    chosen.sort(key=lambda x: x[0], reverse=True)
                    picked = picked[:-len(chosen)] + chosen
        # garder un ordre global "meilleur d'abord"
        picked.sort(key=lambda x: x[0], reverse=True)

    # Mémoriser les résultats proposés pour éviter qu'ils reviennent tout de suite
    if cooldown_sec > 0 and picked:
        reco_remember([(it.server_id, it.rating_key) for _, it in picked], ttl_sec=cooldown_sec)


    
    out: List[Dict[str, Any]] = []
    for s, it in picked:
        # Explications simples (optionnelles)
        why: List[str] = []

        try:
            simv = float(getattr(it, "sim", 0.0) or 0.0)
            if simv >= 0.20:
                why.append(f"Proche de tes goûts (sim {simv:.2f})")
        except Exception:
            pass

        try:
            tg = top_genres[:3]
            gmatch = [g for g in (it.genres or []) if g in tg]
            if gmatch:
                why.append("Genres proches: " + ", ".join(gmatch[:2]))
        except Exception:
            pass

        try:
            if it.rating_imdb and it.rating_imdb >= 7.5:
                why.append(f"Bien noté IMDb ({it.rating_imdb:.1f})")
            elif it.rating_tmdb and it.rating_tmdb >= 7.5:
                why.append(f"Bien noté TMDB ({it.rating_tmdb:.1f})")
            elif it.rating_tvdb and it.rating_tvdb >= 7.5:
                why.append(f"Bien noté TVDB ({it.rating_tvdb:.1f})")
        except Exception:
            pass

        # Deep link to Plex Web (open details page on the right server)
        plex_url = None
        try:
            rk = str(it.rating_key)
            key_path = f"/library/metadata/{rk}"
            key_q = quote(key_path, safe="")
            base = (it.base_url or "").rstrip("/")
            mid = (it.machine_id or "").strip()
            if not mid:
                tok = (server_tokens.get(it.server_id) or "").strip()
                mid = (_get_server_machine_id(it.server_id, base, tok) or "").strip()
            if base:
                if mid:
                    plex_url = f"{base}/web/index.html#!/server/{mid}/details?key={key_q}"
                else:
                    # fallback (works on many setups even without machine_id)
                    plex_url = f"{base}/web/index.html#!/details?key={key_q}"
        except Exception:
            plex_url = None

        out.append({
            "server_id": it.server_id,
            "rating_key": it.rating_key,
            "plex_url": plex_url,
            "thumb": it.thumb,
            "server": it.server,
            "library": it.library,
            "type": it.type,
            "title": it.title,
            "year": it.year,
            "duration_min": it.duration_min,
            "rating": it.rating,
            "audience_rating": it.audience_rating,
            "rating_imdb": it.rating_imdb,
            "rating_tmdb": it.rating_tmdb,
            "rating_tvdb": it.rating_tvdb,
            "stars_imdb": _stars_str(_to_stars(it.rating_imdb)),
            "stars_tmdb": _stars_str(_to_stars(it.rating_tmdb)),
            "stars_tvdb": _stars_str(_to_stars(it.rating_tvdb)),
            "stars_plex": _to_stars_5(it.rating if it.rating is not None else it.audience_rating),
            "stars_plex_str": _stars_str(_to_stars_5(it.rating if it.rating is not None else it.audience_rating)),
            "watched": it.watched,
            "genres": it.genres,
            "summary": it.summary,
            "score": round(float(s), 3),
            "sim": getattr(it, "sim", None),
            "why": why,
            "tmdb_id": it.tmdb_id,
            "imdb_id": it.imdb_id,
            "tvdb_id": it.tvdb_id,
        })

    return {
        "ok": True,
        "count": len(out),
        "top_genres_profile": top_genres,
        "engine": "tfidf+ratings+tmdb" if tmdb_key else "tfidf+ratings",
        "results": out,
    }
