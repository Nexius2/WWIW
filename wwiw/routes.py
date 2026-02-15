from __future__ import annotations

import json
from typing import List, Optional

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
import requests

from .config import APP_NAME, DEFAULT_LIMIT, VERSION
from .db import exec_sql, q_all, q_one, meta_get, meta_set
from .plex import scan_all  # /scan utilise ça
from .reco import recommend
from .scan_manager import maybe_enqueue_scan_debounced
from .state import state
from .utils import form_float, form_int, format_ts
from .i18n import build_translator, available_languages


router = APIRouter()
templates = Jinja2Templates(directory="templates")


def _base_ctx(request: Request, mode: str):
    pref = (meta_get("app_language") or "auto").strip() or "auto"

    t, lang_code = build_translator(request, pref)

    return {
        "request": request,
        "app_name": APP_NAME,
        "version": VERSION,
        "mode": mode,
        "t": t,
        "lang_code": lang_code,
        "lang_pref": pref,
        "available_langs": available_languages(),
        "last_scan": format_ts(state.get("last_scan")),
        "scan_running": bool(state.get("scan_running")),
        "error": state.get("last_error"),
    }




@router.get("/", response_class=HTMLResponse)
def root():
    return RedirectResponse("/reco", status_code=302)

@router.get("/scan-status")
def scan_status():
    return JSONResponse({
        "scan_running": bool(state.get("scan_running")),
        "last_scan": state.get("last_scan"),
        "error": state.get("last_error"),
        "pct": int(state.get("progress_pct") or 0),
        "step": state.get("progress_step") or "",
        "detail": state.get("progress_detail") or "",
        "done": int(state.get("progress_done") or 0),
        "total": int(state.get("progress_total") or 0),
    })

# ────────────────────────────────────────────────────────────────
# SERVERS
# ────────────────────────────────────────────────────────────────

@router.get("/servers", response_class=HTMLResponse)
def servers_page(request: Request):
    servers = q_all("SELECT * FROM servers ORDER BY name ASC")

    ctx = _base_ctx(request, mode="servers")
    ctx.update({"servers": servers})
    return templates.TemplateResponse("servers.html", ctx)



# ────────────────────────────────────────────────────────────────
# SETTINGS
# ────────────────────────────────────────────────────────────────

@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    tmdb_api_key = meta_get("tmdb_api_key") or ""
    tmdb_language = meta_get("tmdb_language") or "fr-FR"
    tmdb_region = meta_get("tmdb_region") or "FR"
    auto_scan_hours = meta_get("auto_scan_hours") or "24"
    reco_cooldown_sec = meta_get("reco_cooldown_sec") or "120"

    ctx = _base_ctx(request, mode="settings")
    ctx.update({
        "tmdb_api_key": tmdb_api_key,
        "tmdb_language": tmdb_language,
        "tmdb_region": tmdb_region,
        "auto_scan_hours": auto_scan_hours,
        "reco_cooldown_sec": reco_cooldown_sec,
    })
    return templates.TemplateResponse("settings.html", ctx)

@router.post("/settings/lang")
def language_settings_save(app_language: str = Form("auto")):
    v = (app_language or "auto").strip() or "auto"

    if v != "auto":
        langs = set(available_languages())
        if v not in langs:
            base = v.split("-", 1)[0]
            if base in langs:
                v = base
            else:
                v = "auto"

    meta_set("app_language", v)
    return RedirectResponse("/settings", status_code=303)


@router.post("/settings/tmdb")
def tmdb_settings_save(
    api_key: str = Form(""),
    language: str = Form("fr-FR"),
    region: str = Form("FR"),
):
    meta_set("tmdb_api_key", (api_key or "").strip())
    meta_set("tmdb_language", (language or "fr-FR").strip())
    meta_set("tmdb_region", (region or "FR").strip())
    return RedirectResponse("/settings", status_code=303)

@router.post("/settings/scan")
def scan_settings_save(auto_scan_hours: str = Form("24")):
    v = (auto_scan_hours or "").strip()
    if not v:
        v = "24"
    try:
        _ = int(float(v))  # accepte 0 et valeurs positives
    except Exception:
        v = "24"
    meta_set("auto_scan_hours", v)
    return RedirectResponse("/settings", status_code=303)

@router.post("/settings/reco")
def reco_settings_save(reco_cooldown_sec: str = Form("120")):
    v = (reco_cooldown_sec or "").strip()
    if not v:
        v = "120"
    try:
        vv = int(float(v))
        if vv < 0:
            vv = 0
        v = str(vv)
    except Exception:
        v = "120"
    meta_set("reco_cooldown_sec", v)
    return RedirectResponse("/settings", status_code=303)


@router.post("/scan-now")
def scan_now():
    # Lance le scan en background (non-bloquant)
    try:
        # évite de lancer si déjà en cours
        if not state.get("scan_running"):
            # thread daemon => la page répond immédiatement
            from .plex import trigger_scan_async
            trigger_scan_async()
        else:
            # déjà en cours => pas d'erreur
            state["last_error"] = None
    except Exception as e:
        state["last_error"] = str(e) or "Scan failed"

    # IMPORTANT: on redirige tout de suite => UI utilisable
    return RedirectResponse("/settings", status_code=303)



@router.post("/servers/add")
def servers_add(
    name: str = Form(...),
    base_url: str = Form(...),
    token: str = Form(...),
):
    name = (name or "").strip()
    base_url = (base_url or "").strip()
    token = (token or "").strip()

    if not name or not base_url or not token:
        return RedirectResponse("/servers", status_code=303)

    exec_sql(
        "INSERT INTO servers(name, base_url, token, enabled, created_at) VALUES(?,?,?,?,?)",
        (name, base_url, token, 1, __import__("datetime").datetime.utcnow().isoformat() + "Z"),
    )

    # ✅ scan auto après ajout (debounce / anti-multi-ajout)
    maybe_enqueue_scan_debounced()

    return RedirectResponse("/servers", status_code=303)


@router.post("/servers/{server_id}/update")
def servers_update(
    server_id: int,
    name: str = Form(...),
    base_url: str = Form(...),
    token: str = Form(""),  # optionnel
):
    name = (name or "").strip()
    base_url = (base_url or "").strip()
    token = (token or "").strip()

    if not name or not base_url:
        return RedirectResponse("/servers", status_code=303)

    exec_sql(
        "UPDATE servers SET name=?, base_url=? WHERE id=?",
        (name, base_url, server_id),
    )

    if token:
        exec_sql(
            "UPDATE servers SET token=? WHERE id=?",
            (token, server_id),
        )

    # ✅ optionnel : rescanner après édition (debounce)
    maybe_enqueue_scan_debounced()

    return RedirectResponse("/servers", status_code=303)


@router.post("/servers/{server_id}/toggle")
def servers_toggle(server_id: int):
    row = q_one("SELECT enabled FROM servers WHERE id=?", (server_id,))
    if row is not None:
        enabled = 0 if int(row["enabled"]) == 1 else 1
        exec_sql("UPDATE servers SET enabled=? WHERE id=?", (enabled, server_id))

        # optionnel : un changement serveur peut impacter la reco (si tu veux)
        # maybe_enqueue_scan_debounced()

    return RedirectResponse("/servers", status_code=303)


@router.post("/servers/{server_id}/delete")
def servers_delete(server_id: int):
    exec_sql("DELETE FROM servers WHERE id=?", (server_id,))
    return RedirectResponse("/servers", status_code=303)


# ────────────────────────────────────────────────────────────────
# LIBRARIES
# ────────────────────────────────────────────────────────────────

@router.get("/libraries", response_class=HTMLResponse)
def libraries_page(request: Request):
    rows = q_all("""
        SELECT l.*, s.name as server_name
        FROM libraries l
        JOIN servers s ON s.id=l.server_id
        ORDER BY s.name ASC, l.type ASC, l.title ASC
    """)
    servers = q_all("SELECT * FROM servers ORDER BY name ASC")

    ctx = _base_ctx(request, mode="libraries")
    ctx.update({
        "libraries": rows,
        "servers": servers,
        "reco": None,
        "filters": None,
        "genres": [],
        "default_limit": DEFAULT_LIMIT,
    })
    return templates.TemplateResponse("reco.html", ctx)




@router.post("/libraries/{lib_id}/toggle")
def libraries_toggle(lib_id: int):
    row = q_one("SELECT enabled FROM libraries WHERE id=?", (lib_id,))
    if row is not None:
        enabled = 0 if int(row["enabled"]) == 1 else 1
        exec_sql("UPDATE libraries SET enabled=? WHERE id=?", (enabled, lib_id))
    # ✅ tu étais sur /reco, mais tu es sur l'onglet libraries
    return RedirectResponse("/libraries", status_code=303)


# ────────────────────────────────────────────────────────────────
# SCAN ENDPOINT 
# ────────────────────────────────────────────────────────────────

@router.post("/scan")
def scan():
    res = scan_all()
    # compat UI
    if res.get("ok") and res.get("last_scan"):
        res["last_scan"] = format_ts(res["last_scan"])
    return JSONResponse(res)


# ────────────────────────────────────────────────────────────────
# RECO UI
# ────────────────────────────────────────────────────────────────

@router.get("/reco", response_class=HTMLResponse)
def reco_page(request: Request):
    libs = q_all("""
        SELECT l.id, l.title, l.type, l.enabled, s.name as server_name
        FROM libraries l
        JOIN servers s ON s.id = l.server_id
        WHERE s.enabled = 1
        ORDER BY s.name ASC, l.type ASC, l.title ASC
    """)

    g_rows = q_all("SELECT genres_json FROM items LIMIT 4000")
    genres_set = set()
    for r in g_rows:
        try:
            gs = json.loads(r["genres_json"] or "[]")
        except Exception:
            gs = []
        for g in gs:
            genres_set.add(g)
    genres = sorted(genres_set)

    ctx = _base_ctx(request, mode="reco")
    ctx.update({
        "libraries": libs,
        "reco": None,
        "filters": None,
        "genres": genres,
        "default_limit": DEFAULT_LIMIT,
    })
    return templates.TemplateResponse("reco.html", ctx)


@router.get("/poster")
def poster(server_id: int, thumb: str):
    srv = q_one("SELECT base_url, token FROM servers WHERE id=? AND enabled=1", (server_id,))
    if not srv:
        return Response(status_code=404)

    base_url = str(srv["base_url"]).rstrip("/")
    token = str(srv["token"] or "").strip()

    if not thumb or not thumb.startswith("/"):
        return Response(status_code=400)

    # URL Plex complète
    url = f"{base_url}{thumb}"

    # Ajoute aussi le token en query param (Plex accepte très bien ça)
    if token:
        if "X-Plex-Token=" not in url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}X-Plex-Token={token}"

    try:
        r = requests.get(
            url,
            headers={"X-Plex-Token": token} if token else {},
            timeout=15,
            allow_redirects=True,
            verify=False,  # IMPORTANT: Plex en HTTPS self-signed => sinon ça plante en requests
        )

        if r.status_code != 200:
            return Response(status_code=r.status_code)

        ctype = r.headers.get("Content-Type") or "image/jpeg"
        return Response(content=r.content, media_type=ctype)

    except Exception:
        return Response(status_code=502)



@router.post("/reco", response_class=HTMLResponse)
def reco_post(
    request: Request,
    type: str = Form("both"),
    limit: int = Form(DEFAULT_LIMIT),
    include_watched: Optional[str] = Form(None),
    max_age_years: Optional[str] = Form(None),
    min_rating: Optional[str] = Form(None),
    min_stars: Optional[str] = Form(None),
    duration_min: Optional[str] = Form(None),
    duration_max: Optional[str] = Form(None),
    libraries: Optional[List[str]] = Form(None),
    include_genres: Optional[List[str]] = Form(None),
    exclude_genres: Optional[List[str]] = Form(None),
):
    filters = {
        "type": type,
        "limit": limit,
        "include_watched": include_watched == "on",
        "max_age_years": form_int(max_age_years),
        "min_rating": form_float(min_rating),
        "min_stars": form_float(min_stars),
        "duration_min": form_int(duration_min),
        "duration_max": form_int(duration_max),
        "libraries_ids": [str(x).strip() for x in (libraries or []) if str(x).strip()],
        "include_genres": [g.strip() for g in (include_genres or [])],
        "exclude_genres": [g.strip() for g in (exclude_genres or [])],
    }

    res = recommend(filters)

    libs = q_all("""
        SELECT l.id, l.title, l.type, l.enabled, s.name as server_name
        FROM libraries l
        JOIN servers s ON s.id=l.server_id
        WHERE s.enabled=1
        ORDER BY s.name ASC, l.type ASC, l.title ASC
    """)

    g_rows = q_all("SELECT genres_json FROM items LIMIT 4000")
    genres_set = set()
    for r in g_rows:
        try:
            gs = json.loads(r["genres_json"] or "[]")
        except Exception:
            gs = []
        for g in gs:
            genres_set.add(g)
    genres = sorted(genres_set)

    ctx = _base_ctx(request, mode="reco")
    ctx.update({
        "libraries": libs,
        "error": state.get("last_error") or (None if res.get("ok") else res.get("error")),
        "reco": res if res.get("ok") else None,
        "filters": filters,
        "genres": genres,
        "default_limit": DEFAULT_LIMIT,
    })
    return templates.TemplateResponse("reco.html", ctx)

