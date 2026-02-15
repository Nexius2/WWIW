from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from plexapi.server import PlexServer

from .config import MIN_SECONDS_BETWEEN_SCANS
from .db import db, q_all, meta_set
from .state import scan_lock, state
from .utils import (
    _duration_min,
    _safe_float,
    _safe_int,
    _tags,
    extract_provider_ratings,
)

def connect_plex(base_url: str, token: str) -> PlexServer:
    return PlexServer(base_url, token)

def _get_section_type(section) -> str:
    st = getattr(section, "type", None) or ""
    return "movie" if st == "movie" else ("show" if st == "show" else st)

def ensure_full_metadata(m):
    try:
        m.reload()
    except Exception:
        pass
    return m

def _extract_external_ids(media) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Extrait imdb_id/tmdb_id/tvdb_id depuis Plex (Guid / guids).

    Avec le nouvel agent Plex, les IDs sont souvent dans une liste de Guid:
      imdb://tt0054215, tmdb://539, tvdb://12345
    """
    imdb_id: Optional[str] = None
    tmdb_id: Optional[int] = None
    tvdb_id: Optional[int] = None

    guids = []
    try:
        guids = list(getattr(media, "guids", None) or [])
    except Exception:
        guids = []

    guid_single = None
    try:
        guid_single = getattr(media, "guid", None)
    except Exception:
        guid_single = None
    if guid_single:
        guids.append(guid_single)

    for g in guids:
        try:
            gid = getattr(g, "id", None) if not isinstance(g, str) else g
            s = str(gid or "")
        except Exception:
            continue
        s = s.strip()
        if not s:
            continue

        low = s.lower()
        if low.startswith("imdb://") and imdb_id is None:
            imdb_id = s.split("://", 1)[1].strip()
        elif low.startswith("tmdb://") and tmdb_id is None:
            try:
                tmdb_id = int(s.split("://", 1)[1].strip())
            except Exception:
                pass
        elif low.startswith("tvdb://") and tvdb_id is None:
            try:
                tvdb_id = int(s.split("://", 1)[1].strip())
            except Exception:
                pass

    return imdb_id, tmdb_id, tvdb_id

def trigger_scan_async():
    if state.get("scan_running"):
        return

    # petit flag immédiat pour que l'UI l'affiche dès le redirect
    state["scan_running"] = True
    state["last_error"] = None

    def _run():
        try:
            res = scan_all()
            if not res.get("ok"):
                state["last_error"] = res.get("error") or "Scan failed"
        except Exception as e:
            state["last_error"] = str(e)
        finally:
            # si scan_all() a échoué avant de remettre scan_running à False,
            # on sécurise pour éviter un état bloqué.
            state["scan_running"] = bool(state.get("scan_running")) and False

    threading.Thread(target=_run, daemon=True).start()


def scan_all() -> Dict[str, Any]:
    now = time.time()
    if now - state["last_scan_ts"] < MIN_SECONDS_BETWEEN_SCANS:
        return {
            "ok": False,
            "error": f"Scan trop récent. Réessaie dans {int(MIN_SECONDS_BETWEEN_SCANS - (now - state['last_scan_ts']))}s.",
        }

    acquired = scan_lock.acquire(blocking=False)
    if not acquired:
        return {"ok": False, "error": "Scan déjà en cours."}

    # Toujours reset + release dans finally (même si exception / return)
    try:
        state["scan_running"] = True
        state["last_error"] = None

        # init progress
        state["progress_pct"] = 0
        state["progress_step"] = "Préparation…"
        state["progress_detail"] = ""
        state["progress_done"] = 0
        state["progress_total"] = 0

        servers = q_all("SELECT * FROM servers WHERE enabled=1 ORDER BY name ASC")
        if not servers:
            return {"ok": False, "error": "Aucun serveur activé. Va dans l’onglet Servers."}

        total_libs = 0
        total_items = 0
        updated_at = datetime.utcnow().isoformat() + "Z"

        with db() as conn:
            for s in servers:
                server_id = int(s["id"])
                base_url = str(s["base_url"])
                token = str(s["token"])

                plex = connect_plex(base_url, token)

                # Store Plex machine identifier (used for deep links to Plex Web)
                try:
                    mid = getattr(plex, "machineIdentifier", None) or getattr(plex, "machine_id", None)
                    if mid:
                        conn.execute("UPDATE servers SET machine_id=? WHERE id=?", (str(mid), server_id))
                except Exception:
                    pass

                libs = []
                server_est_total = 0

                state["progress_step"] = f"Connexion Plex: {s['name']}"
                state["progress_detail"] = ""

                for sec in plex.library.sections():
                    stype = _get_section_type(sec)
                    if stype in ("movie", "show"):
                        try:
                            sz = int(getattr(sec, "totalSize", 0) or 0)
                        except Exception:
                            sz = 0
                        server_est_total += max(sz, 0)
                        libs.append((server_id, int(sec.key), sec.title, stype, updated_at, sz))

                # total global pour %
                state["progress_total"] = int(state.get("progress_total") or 0) + int(server_est_total or 0)

                for (sid, plex_key, title, stype, upd, sz) in libs:
                    conn.execute(
                        """
                        INSERT INTO libraries(server_id, plex_key, title, type, enabled, updated_at)
                        VALUES(?,?,?,?,1,?)
                        ON CONFLICT(server_id, plex_key) DO UPDATE SET
                          title=excluded.title,
                          type=excluded.type,
                          updated_at=excluded.updated_at
                        """,
                        (sid, plex_key, title, stype, upd),
                    )

                enabled_library_ids = []
                for (sid, plex_key, title, stype, upd, sz) in libs:
                    row = conn.execute(
                        "SELECT id, enabled FROM libraries WHERE server_id=? AND plex_key=?",
                        (sid, plex_key),
                    ).fetchone()
                    if row and int(row["enabled"]) == 1:
                        enabled_library_ids.append(int(row["id"]))

                for (sid, plex_key, title, stype, upd, sz) in libs:
                    row = conn.execute(
                        "SELECT id, enabled FROM libraries WHERE server_id=? AND plex_key=?",
                        (sid, plex_key),
                    ).fetchone()
                    if not row or int(row["enabled"]) != 1:
                        continue

                    lib_id = int(row["id"])
                    state["progress_detail"] = f"Bibliothèque: {title}"

                    section = plex.library.sectionByID(plex_key)

                    try:
                        items = section.all()
                    except Exception:
                        items = []

                    for it in items:
                        try:
                            imdb_id, tmdb_id, tvdb_id = _extract_external_ids(ensure_full_metadata(it))
                            rating_key = str(getattr(it, "ratingKey", None) or "").strip()
                            if not rating_key:
                                continue
                        except Exception:
                            imdb_id, tmdb_id, tvdb_id = (None, None, None)

                        try:
                            duration_min = int((getattr(it, "duration", 0) or 0) / 60000)
                        except Exception:
                            duration_min = None

                        try:
                            rating = float(getattr(it, "rating", None)) if getattr(it, "rating", None) is not None else None
                        except Exception:
                            rating = None

                        try:
                            audience_rating = float(getattr(it, "audienceRating", None)) if getattr(it, "audienceRating", None) is not None else None
                        except Exception:
                            audience_rating = None

                        try:
                            year = int(getattr(it, "year", None)) if getattr(it, "year", None) is not None else None
                        except Exception:
                            year = None

                        try:
                            watched = 1 if getattr(it, "isWatched", False) else 0
                        except Exception:
                            watched = 0


                        # Signals de visionnage (pour reco plus intelligente)
                        last_viewed_at = None
                        view_count = None
                        added_at = None
                        try:
                            lv = getattr(it, "lastViewedAt", None)
                            if lv:
                                try:
                                    last_viewed_at = lv.timestamp()
                                except Exception:
                                    pass
                            view_count = getattr(it, "viewCount", None)
                            ad = getattr(it, "addedAt", None)
                            if ad:
                                try:
                                    added_at = ad.timestamp()
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        # Additional watch signals (optional)
                        user_rating = None
                        view_offset_ms = None
                        duration_ms = None
                        completion_ratio = None
                        try:
                            user_rating = getattr(it, "userRating", None)
                        except Exception:
                            pass
                        try:
                            view_offset_ms = getattr(it, "viewOffset", None)
                        except Exception:
                            pass
                        try:
                            duration_ms = getattr(it, "duration", None)
                        except Exception:
                            pass
                        try:
                            if view_offset_ms is not None and duration_ms:
                                completion_ratio = float(view_offset_ms) / float(duration_ms)
                                # clamp
                                if completion_ratio < 0:
                                    completion_ratio = 0.0
                                if completion_ratio > 1.2:
                                    completion_ratio = 1.2
                        except Exception:
                            completion_ratio = None

                        try:
                            genres = [g.tag for g in (getattr(it, "genres", None) or []) if getattr(g, "tag", None)]
                        except Exception:
                            genres = []

                        try:
                            summary = getattr(it, "summary", None)
                        except Exception:
                            summary = None

                        try:
                            thumb = getattr(it, "thumb", None)
                        except Exception:
                            thumb = None

                        conn.execute(
                            """
                            INSERT INTO items(server_id, library_id, rating_key, type, title, year, duration_min, rating, audience_rating, watched,
                                              imdb_id, tmdb_id, tvdb_id, genres_json, summary, thumb,
                                              last_viewed_at, view_count, added_at,
                                              user_rating, view_offset_ms, duration_ms, completion_ratio,
                                              updated_at)
                            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                            ON CONFLICT(server_id, rating_key) DO UPDATE SET
                              library_id=excluded.library_id,
                              type=excluded.type,
                              title=excluded.title,
                              year=excluded.year,
                              duration_min=excluded.duration_min,
                              rating=excluded.rating,
                              audience_rating=excluded.audience_rating,
                              watched=excluded.watched,
                              imdb_id=excluded.imdb_id,
                              tmdb_id=excluded.tmdb_id,
                              tvdb_id=excluded.tvdb_id,
                              genres_json=excluded.genres_json,
                              summary=excluded.summary,
                              thumb=excluded.thumb,
                              last_viewed_at=excluded.last_viewed_at,
                              view_count=excluded.view_count,
                              added_at=excluded.added_at,
                              user_rating=excluded.user_rating,
                              view_offset_ms=excluded.view_offset_ms,
                              duration_ms=excluded.duration_ms,
                              completion_ratio=excluded.completion_ratio,
                              updated_at=excluded.updated_at
                            """,
                            (
                                server_id, lib_id, rating_key, stype,
                                getattr(it, "title", None),
                                year,
                                duration_min,
                                rating,
                                audience_rating,
                                watched,
                                imdb_id, tmdb_id, tvdb_id,
                                json.dumps(genres, ensure_ascii=False),
                                summary,
                                thumb,
                                last_viewed_at,
                                view_count,
                                added_at,
                                user_rating,
                                view_offset_ms,
                                duration_ms,
                                completion_ratio,
                                updated_at,
                            ),
                        )

                        total_items += 1
                        state["progress_done"] = int(state.get("progress_done") or 0) + 1
                        tot = int(state.get("progress_total") or 0)
                        if tot > 0:
                            state["progress_pct"] = min(99, int((state["progress_done"] * 100) / tot))

                    total_libs += 1

                # purge items anciens uniquement pour libs activées
                conn.execute(
                    "UPDATE servers SET last_scan=? WHERE id=?",
                    (updated_at, server_id),
                )

                if enabled_library_ids:
                    qmarks = ",".join(["?"] * len(enabled_library_ids))
                    conn.execute(
                        f"DELETE FROM items WHERE server_id=? AND updated_at<>? AND library_id IN ({qmarks})",
                        (server_id, updated_at, *enabled_library_ids),
                    )

            conn.commit()

        state["last_scan_ts"] = now
        state["last_scan"] = updated_at

        state["progress_pct"] = 100
        state["progress_step"] = "Terminé"
        state["progress_detail"] = ""

        meta_set("last_scan_ts", str(state["last_scan_ts"]))
        meta_set("last_scan_iso", state["last_scan"] or "")

        return {
            "ok": True,
            "servers": len(servers),
            "libraries_scanned": total_libs,
            "items_scanned": total_items,
            "last_scan": updated_at,
        }

    except Exception as e:
        state["last_error"] = str(e)
        return {"ok": False, "error": str(e)}

    finally:
        state["scan_running"] = False
        try:
            if acquired:
                scan_lock.release()
        except Exception:
            # on ne veut JAMAIS crasher ici
            pass


def auto_scan_loop():
    CHECK_EVERY_SEC = 300  # vérifie toutes les 5 minutes

    while True:
        try:
            # interval en heures stocké en DB (meta)
            try:
                hours = int(float((meta_get("auto_scan_hours") or "24").strip()))
            except Exception:
                hours = 24

            # 0 = désactivé
            if hours <= 0:
                time.sleep(CHECK_EVERY_SEC)
                continue

            threshold_sec = hours * 3600

            now = time.time()
            last_ts = float(state.get("last_scan_ts") or 0.0)
            if last_ts <= 0 or (now - last_ts) >= threshold_sec:
                scan_all()

        except Exception:
            pass

        time.sleep(CHECK_EVERY_SEC)

