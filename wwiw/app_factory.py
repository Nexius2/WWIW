from __future__ import annotations

import threading
from urllib.parse import quote

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates

from .db import init_db, meta_get, meta_set, migrate_db
from .plex import auto_scan_loop
from .routes import router, templates as routes_templates
from .state import state
from fastapi.staticfiles import StaticFiles


def create_app() -> FastAPI:
    app = FastAPI(title="WWIW")
    app.mount("/static", StaticFiles(directory="static"), name="static")

    # IMPORTANT: templates doivent rester sur "templates/" (racine projet),
    # et on garde le filtre urlencode.
    routes_templates.env.filters["urlencode"] = quote

    @app.on_event("startup")
    def _startup():
        init_db()
        migrate_db()
        # Defaults DB (1 seule fois)
        if not (meta_get("auto_scan_hours") or "").strip():
            meta_set("auto_scan_hours", "24")


        try:
            ts = meta_get("last_scan_ts")
            if ts:
                state["last_scan_ts"] = float(ts)
            iso = meta_get("last_scan_iso")
            if iso:
                state["last_scan"] = iso
        except Exception:
            pass

        t = threading.Thread(target=auto_scan_loop, daemon=True)
        t.start()

    app.include_router(router)
    return app
