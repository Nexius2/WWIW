# 🎬 WWIW — What Will I Watch?

**WWIW is a self‑hosted intelligent recommendation engine for Plex power users.**

It connects to one or multiple Plex servers, analyzes your media patterns, and suggests what you are most likely to enjoy next.

Because choosing what to watch shouldn't take longer than watching it.

---

## 🧭 What Is WWIW?

WWIW eliminates endless scrolling through large libraries by generating smart, taste-aware recommendations based on your actual viewing behavior.

It does not randomly shuffle content.
It evaluates patterns, preferences, and context to surface relevant suggestions instantly.

Designed for advanced Plex environments, including multi-server setups.

---

## 🌐 Multi‑Server Plex Support

WWIW is built to handle complex Plex infrastructures.

You can:

- Register multiple Plex servers
- Store independent base URLs and tokens
- Manage multiple libraries per server
- Filter recommendations by server or library
- Merge content across servers into unified suggestions

Ideal for distributed setups or segmented environments.

---

## 🧠 How Recommendations Work

WWIW builds a scoring model from your media data and combines multiple signals:

- Content similarity using TF‑IDF + cosine similarity (synopsis, genres, metadata)
- Genre affinity weighting based on your watch patterns
- Ratings (Plex rating, audience rating, user rating when available)
- Watch status influence (completed vs partial views)
- Optional TMDB trending/popularity boost (via API key)
- Age filtering (year weighting)
- Configurable cooldown logic to prevent repetition

The result:

Recommendations that feel personal — not random.

---

## 🔁 Intelligent Anti‑Repetition

When you click “Generate” again, WWIW avoids repeating the same suggestions immediately.

A configurable cooldown system (`reco_cooldown_sec`, default: 120 seconds) keeps recommendations fresh and varied.

---

## 🎛 Advanced Filtering Controls

WWIW allows fine‑tuned recommendation control:

- Movies, TV shows, or both
- Duration range filtering
- Minimum rating / star filtering
- Maximum age filtering
- Include or exclude genres
- Include or exclude watched content
- Filter per server
- Filter per library

You control the context.  
WWIW handles the scoring.

---

## ⚙️ Architecture

WWIW is built with:

- FastAPI
- SQLite (WAL mode)
- Jinja2
- Docker-ready deployment

All configuration is stored in the internal database — no mandatory environment variables.

---

## 🚀 Installation (Docker — Recommended)

### Requirements
- Docker

### Build & Run

From the project directory:

```bash
./create-container-test.sh
```

Default ports:

- Host: 8788
- Container: 8787

Then open:

```
http://<YOUR_SERVER_IP>:8788
```

---

## 🛠 Manual Installation

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8788
```

FastAPI entry point: `app.py`  
Application factory: `wwiw/app_factory.py`

---

## 🧩 Configuration

All setup is handled inside the web interface:

1. Add one or more Plex servers
2. Launch a scan
3. Adjust filters and settings
4. Generate smart recommendations

Optional:
Add a TMDB API key to enhance scoring with trending/popularity data.

---

## 📌 Technical Notes

- Asynchronous scanning with UI feedback
- Persistent settings stored in SQLite
- Multi-server architecture support
- Optimized for Docker environments
- Designed for large libraries

---

## 🎯 Why WWIW?

Even with terabytes of media, the hardest question remains:

“What should I watch?”

WWIW answers it intelligently.

---

## License

MIT
