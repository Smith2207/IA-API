import json
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np

from app.config import get_settings
from app.db.repository import PetEmbeddingRepository
from app.services.embeddings import ClipEmbeddingService
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

EMBEDDING_DIM = 512  # ViT-B-32


class FaissSearchIndex:
    """Índice vectorial en memoria sincronizado con Neon PostgreSQL."""

    _instance: Optional["FaissSearchIndex"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._index: Optional[faiss.IndexIDMap2] = None
        self._ids: List[str] = []
        self._metadata: dict[str, dict] = {}
        self._faiss_id_map: dict[int, str] = {}
        self._repo = PetEmbeddingRepository()

    @classmethod
    def get_instance(cls) -> "FaissSearchIndex":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_index(self) -> faiss.IndexIDMap2:
        if self._index is None:
            base = faiss.IndexFlatIP(EMBEDDING_DIM)
            self._index = faiss.IndexIDMap2(base)
        return self._index

    @staticmethod
    def _pet_id_to_faiss(pet_id: str) -> int:
        import hashlib

        digest = hashlib.sha256(pet_id.encode("utf-8")).digest()[:8]
        return int.from_bytes(digest, "big") & 0x7FFFFFFFFFFFFFFF

    def rebuild_from_db(self) -> int:
        with self._lock:
            rows = self._repo.list_all()
            base = faiss.IndexFlatIP(EMBEDDING_DIM)
            index = faiss.IndexIDMap2(base)
            self._ids = []
            self._metadata = {}
            self._faiss_id_map: dict[int, str] = {}

            for row in rows:
                emb = np.array(row["embedding"], dtype=np.float32)
                if emb.shape[0] != EMBEDDING_DIM:
                    logger.warning(
                        "Embedding id=%s tiene dimensión %s, se omite",
                        row["id"],
                        emb.shape[0],
                    )
                    continue
                emb = ClipEmbeddingService.normalize(emb)
                faiss_id = self._pet_id_to_faiss(row["id"])
                index.add_with_ids(
                    emb.reshape(1, -1).astype(np.float32),
                    np.array([faiss_id], dtype=np.int64),
                )
                self._ids.append(row["id"])
                self._faiss_id_map[faiss_id] = row["id"]
                self._metadata[row["id"]] = {
                    "pet_name": row["pet_name"],
                    "location": row.get("location") or "",
                    "image_url": row["image_url"],
                    "detected_class": row.get("detected_class"),
                }

            self._index = index
            count = len(self._ids)
            logger.info("Índice FAISS reconstruido con %s vectores", count)
            return count

    def add_or_update(
        self,
        pet_id: str,
        embedding: np.ndarray,
        metadata: dict,
    ) -> None:
        embedding = ClipEmbeddingService.normalize(embedding)
        with self._lock:
            index = self._ensure_index()
            faiss_id = self._pet_id_to_faiss(pet_id)

            if pet_id in self._ids:
                try:
                    index.remove_ids(np.array([faiss_id], dtype=np.int64))
                except Exception:
                    logger.debug("Reindexando entrada existente id=%s", pet_id)
                self._ids = [i for i in self._ids if i != pet_id]

            index.add_with_ids(
                embedding.reshape(1, -1).astype(np.float32),
                np.array([faiss_id], dtype=np.int64),
            )
            self._ids.append(pet_id)
            self._faiss_id_map[faiss_id] = pet_id
            self._metadata[pet_id] = metadata

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
        exclude_id: Optional[str] = None,
    ) -> List[Tuple[str, float, dict]]:
        settings = get_settings()
        k = top_k or settings.search_top_k
        query_embedding = ClipEmbeddingService.normalize(query_embedding)

        with self._lock:
            if self._index is None or len(self._ids) == 0:
                return []

            index = self._index
            fetch_k = min(k + 5, len(self._ids))
            scores, faiss_ids = index.search(
                query_embedding.reshape(1, -1).astype(np.float32),
                fetch_k,
            )

        results: List[Tuple[str, float, dict]] = []
        for score, fid in zip(scores[0], faiss_ids[0]):
            if fid < 0:
                continue
            pet_id = self._faiss_id_map.get(int(fid))
            if not pet_id:
                continue
            if exclude_id and pet_id == exclude_id:
                continue
            similarity_percent = max(0.0, min(100.0, float(score) * 100.0))
            if similarity_percent < settings.similarity_min_percent:
                continue
            meta = self._metadata.get(pet_id, {})
            results.append((pet_id, similarity_percent, meta))
            if len(results) >= k:
                break

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @property
    def vector_count(self) -> int:
        return len(self._ids)

    def persist_snapshot(self, path: Path) -> None:
        """Opcional: guardar ids para depuración (FAISS se reconstruye desde DB)."""
        with self._lock:
            snapshot = {
                "ids": self._ids,
                "metadata": self._metadata,
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
