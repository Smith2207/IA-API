"""Ejecutar desde ia-api/: python scripts/init_db.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.connection import close_pool, init_db, init_pool
from app.utils.logging_config import setup_logging


def main() -> None:
    setup_logging()
    init_pool()
    init_db()
    close_pool()
    print("Tabla pet_embeddings lista.")


if __name__ == "__main__":
    main()
