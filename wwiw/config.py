import os
from pathlib import Path

APP_NAME = "WWIW"


def _read_version_from_info(default: str = "dev") -> str:

    # Emplacements possibles (dev + docker)
    candidates = []

    # Optionnel: permettre de forcer un chemin via env
    info_env = (os.getenv("INFO_PATH") or "").strip()
    if info_env:
        candidates.append(Path(info_env))

    # 1) /app/INFO (docker WORKDIR /app)
    candidates.append(Path("INFO"))

    # 2) INFO à côté du projet (si lancé depuis la racine du repo)
    # config.py est dans /app/wwiw/wwiw/config.py -> parents[2] = /app
    # et parents[1] = /app/wwiw
    here = Path(__file__).resolve()
    candidates.append(here.parents[2] / "INFO")  # /app/INFO
    candidates.append(here.parents[1] / "INFO")  # /app/wwiw/INFO (si tu le laisses là)

    for p in candidates:
        try:
            if p and p.exists():
                for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if line.startswith("VERSION="):
                        return line.split("=", 1)[1].strip()
        except Exception:
            pass

    return default


# ✅ Version dynamique depuis INFO
VERSION = _read_version_from_info()

DATABASE_PATH = os.getenv("DATABASE_PATH", "/appdata/wwiw.db").strip()
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "18"))
MIN_SECONDS_BETWEEN_SCANS = int(os.getenv("MIN_SECONDS_BETWEEN_SCANS", "500"))

# ────────────────────────────────────────────────────────────────
# External signals (optionnels)
# ────────────────────────────────────────────────────────────────
# TMDB est le plus simple pour "tendance/populaire".
# Crée une clé API (v3) et mets-la en variable d'env: TMDB_API_KEY
#TMDB_API_KEY = (os.getenv("TMDB_API_KEY") or "").strip()
TMDB_CACHE_HOURS = int(os.getenv("TMDB_CACHE_HOURS", "12"))
