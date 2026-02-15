from __future__ import annotations

import threading
import time

from .plex import trigger_scan_async
from .state import state

_debounce_lock = threading.Lock()
_debounce_timer: threading.Timer | None = None


def enqueue_scan_if_idle() -> bool:
    """
    Déclenche un scan si aucun scan n'est déjà en cours.
    IMPORTANT: on ne touche PAS à state["scan_running"] ici,
    car c'est scan_all()/plex.py qui le gère correctement.
    """
    if state.get("scan_running"):
        return False

    try:
        trigger_scan_async()
        return True
    except Exception as e:
        state["last_error"] = str(e)
        return False


def maybe_enqueue_scan_debounced(delay_sec: float = 2.0) -> None:
    """
    Debounce: si tu ajoutes/édites plusieurs serveurs rapidement,
    on ne lance qu'un scan après un petit délai.
    """
    global _debounce_timer

    with _debounce_lock:
        if _debounce_timer is not None:
            try:
                _debounce_timer.cancel()
            except Exception:
                pass

        def _run():
            try:
                enqueue_scan_if_idle()
            except Exception:
                pass

        _debounce_timer = threading.Timer(delay_sec, _run)
        _debounce_timer.daemon = True
        _debounce_timer.start()



