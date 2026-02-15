import os
import sqlite3
from typing import List, Optional

from .config import DATABASE_PATH


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db() -> None:
    """
    IMPORTANT:
    - init_db doit être SAFE même si la DB existe déjà avec un ancien schéma.
    - Donc: pas d'index sur des colonnes qui pourraient ne pas exister encore.
    """
    dirpath = os.path.dirname(DATABASE_PATH)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS servers (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL,
              base_url TEXT NOT NULL,
              token TEXT NOT NULL,
              machine_id TEXT,
              enabled INTEGER NOT NULL DEFAULT 1,
              created_at TEXT NOT NULL,
              last_scan TEXT,
              scan_error TEXT
            );

            CREATE TABLE IF NOT EXISTS libraries (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              server_id INTEGER NOT NULL,
              plex_key INTEGER NOT NULL,
              title TEXT NOT NULL,
              type TEXT NOT NULL,             -- movie/show
              enabled INTEGER NOT NULL DEFAULT 1,
              updated_at TEXT NOT NULL,
              UNIQUE(server_id, plex_key),
              FOREIGN KEY(server_id) REFERENCES servers(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS items (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              server_id INTEGER NOT NULL,
              library_id INTEGER NOT NULL,
              rating_key TEXT NOT NULL,
              type TEXT NOT NULL,             -- movie/show
              title TEXT NOT NULL,
              year INTEGER,
              duration_min INTEGER,
              genres_json TEXT,
              rating REAL,
              audience_rating REAL,
              content_rating TEXT,
              watched INTEGER NOT NULL DEFAULT 0,
              summary TEXT,

              -- Watch history signals
              last_viewed_at REAL,            -- epoch seconds
              view_count INTEGER,
              added_at REAL,                  -- epoch seconds
              user_rating REAL,               -- Plex userRating (0-10)
              view_offset_ms INTEGER,         -- Plex viewOffset (ms)
              duration_ms INTEGER,            -- Plex duration (ms)
              completion_ratio REAL,          -- view_offset_ms / duration_ms

              -- External ids (peuvent être ajoutés plus tard via migrate_db)
              imdb_id TEXT,
              tmdb_id INTEGER,
              tvdb_id INTEGER,

              -- Ratings provider
              rating_imdb REAL,
              rating_tmdb REAL,
              rating_tvdb REAL,

              thumb TEXT,
              updated_at TEXT NOT NULL,

              UNIQUE(server_id, rating_key),
              FOREIGN KEY(server_id) REFERENCES servers(id) ON DELETE CASCADE,
              FOREIGN KEY(library_id) REFERENCES libraries(id) ON DELETE CASCADE
            );

            -- Index SAFE (colonnes "anciennes" qui existent quasiment toujours)
            CREATE INDEX IF NOT EXISTS idx_items_type ON items(type);
            CREATE INDEX IF NOT EXISTS idx_items_year ON items(year);
            CREATE INDEX IF NOT EXISTS idx_items_watched ON items(watched);
            CREATE INDEX IF NOT EXISTS idx_items_library ON items(library_id);

            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            """
        )

def migrate_db() -> None:
    """Évolutions du schéma (safe, idempotent)."""
    with db() as conn:

        def table_cols(table: str) -> List[str]:
            try:
                return [r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
            except Exception:
                return []

        def add_col(table: str, cols: List[str], col: str, ddl: str):
            if col in cols:
                return
            conn.execute(ddl)
            conn.commit()
            cols.append(col)

        # ───────────────────────────────
        # servers
        # ───────────────────────────────
        s_cols = table_cols("servers")
        if s_cols:
            add_col("servers", s_cols, "last_scan", "ALTER TABLE servers ADD COLUMN last_scan TEXT;")
            add_col("servers", s_cols, "scan_error", "ALTER TABLE servers ADD COLUMN scan_error TEXT;")
            add_col("servers", s_cols, "machine_id", "ALTER TABLE servers ADD COLUMN machine_id TEXT;")

        # ───────────────────────────────
        # items
        # ───────────────────────────────
        cols = table_cols("items")
        if not cols:
            return

        # Colonnes historiques
        add_col("items", cols, "thumb", "ALTER TABLE items ADD COLUMN thumb TEXT;")

        # Ratings providers
        add_col("items", cols, "rating_imdb", "ALTER TABLE items ADD COLUMN rating_imdb REAL;")
        add_col("items", cols, "rating_tmdb", "ALTER TABLE items ADD COLUMN rating_tmdb REAL;")
        add_col("items", cols, "rating_tvdb", "ALTER TABLE items ADD COLUMN rating_tvdb REAL;")

        # External ids
        add_col("items", cols, "imdb_id", "ALTER TABLE items ADD COLUMN imdb_id TEXT;")
        add_col("items", cols, "tmdb_id", "ALTER TABLE items ADD COLUMN tmdb_id INTEGER;")
        add_col("items", cols, "tvdb_id", "ALTER TABLE items ADD COLUMN tvdb_id INTEGER;")

        # Watch history signals
        add_col("items", cols, "last_viewed_at", "ALTER TABLE items ADD COLUMN last_viewed_at REAL;")
        add_col("items", cols, "view_count", "ALTER TABLE items ADD COLUMN view_count INTEGER;")
        add_col("items", cols, "added_at", "ALTER TABLE items ADD COLUMN added_at REAL;")
        add_col("items", cols, "user_rating", "ALTER TABLE items ADD COLUMN user_rating REAL;")
        add_col("items", cols, "view_offset_ms", "ALTER TABLE items ADD COLUMN view_offset_ms INTEGER;")
        add_col("items", cols, "duration_ms", "ALTER TABLE items ADD COLUMN duration_ms INTEGER;")
        add_col("items", cols, "completion_ratio", "ALTER TABLE items ADD COLUMN completion_ratio REAL;")

        # Index (uniquement si colonnes présentes)
        try:
            if "tmdb_id" in cols:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_items_tmdb_id ON items(tmdb_id);")
            if "imdb_id" in cols:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_items_imdb_id ON items(imdb_id);")
            if "tvdb_id" in cols:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_items_tvdb_id ON items(tvdb_id);")
            if "last_viewed_at" in cols:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_items_last_viewed ON items(last_viewed_at);")
            conn.commit()
        except Exception:
            pass



def q_all(sql: str, params: tuple = ()) -> List[sqlite3.Row]:
    with db() as conn:
        cur = conn.execute(sql, params)
        return list(cur.fetchall())


def q_one(sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    with db() as conn:
        cur = conn.execute(sql, params)
        return cur.fetchone()


def exec_sql(sql: str, params: tuple = ()) -> None:
    with db() as conn:
        conn.execute(sql, params)
        conn.commit()


def meta_get(key: str) -> Optional[str]:
    row = q_one("SELECT value FROM meta WHERE key=?", (key,))
    return str(row["value"]) if row else None


def meta_set(key: str, value: str) -> None:
    with db() as conn:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        conn.commit()