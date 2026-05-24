from contextlib import contextmanager
from typing import Generator, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import get_settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

_pool: Optional[ConnectionPool] = None

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pet_embeddings (
    id TEXT PRIMARY KEY,
    pet_name TEXT NOT NULL,
    location TEXT,
    image_url TEXT NOT NULL,
    embedding JSONB NOT NULL,
    detected_class TEXT,
    bbox JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pet_embeddings_updated
    ON pet_embeddings (updated_at DESC);
"""


def init_pool() -> None:
    global _pool
    if _pool is not None:
        return
    settings = get_settings()
    logger.info("Inicializando pool de PostgreSQL (Neon)")
    _pool = ConnectionPool(
        conninfo=settings.database_url,
        min_size=1,
        max_size=10,
        kwargs={"row_factory": dict_row},
        open=True,
    )


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
        logger.info("Pool de PostgreSQL cerrado")


def init_db() -> None:
    init_pool()
    with get_connection() as conn:
        conn.execute(SCHEMA_SQL)
        conn.commit()
    logger.info("Esquema pet_embeddings verificado")


@contextmanager
def get_connection() -> Generator[psycopg.Connection, None, None]:
    if _pool is None:
        init_pool()
    assert _pool is not None
    with _pool.connection() as conn:
        yield conn
