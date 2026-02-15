import threading
import time
from typing import Any, Dict, Iterable, Set, Tuple


scan_lock = threading.Lock()
state: Dict[str, Any] = {
    "last_scan": None,
    "last_scan_ts": 0.0,
    "scan_running": False,
    "last_error": None,

    # Progress UI
    "progress_pct": 0,
    "progress_step": "",      
    "progress_detail": "",    
    "progress_done": 0,
    "progress_total": 0,
}

_reco_lock = threading.Lock()
_reco_recent: Dict[Tuple[int, str], float] = {}


def reco_recent_keys(now: float | None = None) -> Set[Tuple[int, str]]:
    """Retourne l'ensemble des clés encore sous cooldown et purge le reste."""
    if now is None:
        now = time.time()
    with _reco_lock:
        expired = [k for k, exp in _reco_recent.items() if exp <= now]
        for k in expired:
            _reco_recent.pop(k, None)
        return set(_reco_recent.keys())


def reco_remember(keys: Iterable[Tuple[int, str]], ttl_sec: int, now: float | None = None) -> None:
    """Mémorise des clés pour les exclure pendant ttl_sec."""
    if now is None:
        now = time.time()
    ttl = max(0, int(ttl_sec or 0))
    if ttl <= 0:
        return
    exp = now + ttl
    with _reco_lock:
        for k in keys:
            try:
                sid, rk = k
                _reco_recent[(int(sid), str(rk))] = exp
            except Exception:
                pass
