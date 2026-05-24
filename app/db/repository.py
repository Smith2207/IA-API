from typing import Any, List, Optional

import numpy as np

from app.db.connection import get_connection
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class PetEmbeddingRepository:
    def upsert(
        self,
        pet_id: str,
        pet_name: str,
        location: Optional[str],
        image_url: str,
        embedding: np.ndarray,
        detected_class: Optional[str],
        bbox: Optional[List[float]],
    ) -> None:
        embedding_list = embedding.astype(np.float32).tolist()
        bbox_json = bbox if bbox else None

        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO pet_embeddings (
                    id, pet_name, location, image_url, embedding,
                    detected_class, bbox, updated_at
                )
                VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, NOW())
                ON CONFLICT (id) DO UPDATE SET
                    pet_name = EXCLUDED.pet_name,
                    location = EXCLUDED.location,
                    image_url = EXCLUDED.image_url,
                    embedding = EXCLUDED.embedding,
                    detected_class = EXCLUDED.detected_class,
                    bbox = EXCLUDED.bbox,
                    updated_at = NOW()
                """,
                (
                    pet_id,
                    pet_name,
                    location,
                    image_url,
                    embedding_list,
                    detected_class,
                    bbox_json,
                ),
            )
            conn.commit()
        logger.info("Embedding guardado para mascota id=%s", pet_id)

    def delete(self, pet_id: str) -> bool:
        with get_connection() as conn:
            cur = conn.execute(
                "DELETE FROM pet_embeddings WHERE id = %s RETURNING id",
                (pet_id,),
            )
            row = cur.fetchone()
            conn.commit()
        return row is not None

    def get_by_id(self, pet_id: str) -> Optional[dict[str, Any]]:
        with get_connection() as conn:
            cur = conn.execute(
                """
                SELECT id, pet_name, location, image_url, embedding,
                       detected_class, bbox, created_at, updated_at
                FROM pet_embeddings WHERE id = %s
                """,
                (pet_id,),
            )
            return cur.fetchone()

    def list_all(self) -> List[dict[str, Any]]:
        with get_connection() as conn:
            cur = conn.execute(
                """
                SELECT id, pet_name, location, image_url, embedding,
                       detected_class, bbox
                FROM pet_embeddings
                ORDER BY updated_at DESC
                """
            )
            return list(cur.fetchall())

    def count(self) -> int:
        with get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) AS total FROM pet_embeddings")
            row = cur.fetchone()
        return int(row["total"]) if row else 0
