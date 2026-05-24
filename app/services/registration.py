from typing import Optional

import numpy as np
from PIL import Image

from app.db.repository import PetEmbeddingRepository
from app.services.pipeline import PetVisionPipeline
from app.services.search import FaissSearchIndex
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class PetRegistrationService:
    def __init__(self) -> None:
        self._pipeline = PetVisionPipeline()
        self._repo = PetEmbeddingRepository()
        self._index = FaissSearchIndex.get_instance()

    def register(
        self,
        pet_id: str,
        pet_name: str,
        location: Optional[str],
        image_url: str,
        image: Image.Image,
    ) -> dict:
        detection, embedding = self._pipeline.process_image(image, require_pet=True)

        bbox_list = list(detection.bbox) if detection.bbox else None

        self._repo.upsert(
            pet_id=pet_id,
            pet_name=pet_name,
            location=location,
            image_url=image_url,
            embedding=embedding,
            detected_class=detection.class_name,
            bbox=bbox_list,
        )

        self._index.add_or_update(
            pet_id=pet_id,
            embedding=embedding,
            metadata={
                "pet_name": pet_name,
                "location": location or "",
                "image_url": image_url,
                "detected_class": detection.class_name,
            },
        )

        return {
            "pet_id": pet_id,
            "pet_name": pet_name,
            "detected_class": detection.class_name,
            "confidence": round(detection.confidence * 100, 2),
            "embedding_dim": int(embedding.shape[0]),
        }
