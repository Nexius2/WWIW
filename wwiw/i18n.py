import json
from pathlib import Path
from fastapi import Request
import os

def _find_lang_dir() -> Path:
    # 1) variable d'env (si tu veux un jour la fixer)
    env = os.getenv("WWIW_LANG_DIR")
    if env:
        p = Path(env)
        if p.exists() and p.is_dir():
            return p

    # 2) Cherche en remontant depuis ce fichier
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        cand = parent / "lang"
        if cand.exists() and cand.is_dir():
            return cand

    # 3) fallback final (ne devrait pas arriver)
    return here.parent / "lang"

LANG_DIR = _find_lang_dir()
DEFAULT_LANG = "en"


def available_languages():
    langs = []
    if LANG_DIR.exists():
        for f in LANG_DIR.glob("*.json"):
            langs.append(f.stem)
    return sorted(langs)


def load_lang(lang_code: str):
    path = LANG_DIR / f"{lang_code}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def detect_browser_lang(request: Request):
    header = request.headers.get("accept-language", "")
    if not header:
        return DEFAULT_LANG

    langs = available_languages()
    for part in header.split(","):
        code = part.split(";")[0].strip().lower()
        short = code.split("-")[0]

        if short in langs:
            return short

    return DEFAULT_LANG


def get_lang(request: Request, db_lang: str | None):
    langs = available_languages()

    # 1️⃣ Si langue définie en base
    if db_lang and db_lang != "auto" and db_lang in langs:
        return db_lang

    # 2️⃣ Sinon → langue navigateur
    browser_lang = detect_browser_lang(request)
    if browser_lang in langs:
        return browser_lang

    # 3️⃣ Fallback final
    return DEFAULT_LANG


def build_translator(request: Request, db_lang: str | None):
    lang_code = get_lang(request, db_lang)

    base_dict = load_lang(DEFAULT_LANG)
    lang_dict = load_lang(lang_code)

    def t(key: str, **kwargs):
        # 1) Trouve la traduction (langue choisie -> fallback default -> fallback clé)
        if key in lang_dict:
            text = lang_dict[key]
        elif key in base_dict:
            text = base_dict[key]
        else:
            text = key

        # 2) Si le template passe des variables (count=..., last_scan=...), on formate
        if kwargs:
            try:
                text = text.format(**kwargs)
            except Exception:
                # si la string contient des {xxx} incohérents, on évite de crasher
                pass

        return text


    return t, lang_code
