from typing import Optional, Tuple

import numpy as np
from PIL import Image

from app.exceptions import PetNotDetectedError
from app.services.detection import DetectionResult, PetDetectionService
from app.services.embeddings import ClipEmbeddingService
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class PetVisionPipeline:
    """Orquesta detección YOLO + embedding CLIP."""

    def __init__(self) -> None:
        self._detector = PetDetectionService.get_instance()
        self._embedder = ClipEmbeddingService.get_instance()

    def process_image(
        self,
        image: Image.Image,
        *,
        require_pet: bool = True,
    ) -> Tuple[DetectionResult, np.ndarray]:
        detection = self._detector.detect(image)

        if require_pet and not detection.detected:
            raise PetNotDetectedError(
                "No se detectó un perro o gato en la imagen. "
                "Sube una foto donde la mascota sea claramente visible."
            )

        crop = detection.crop
        if crop.width < 32 or crop.height < 32:
            crop = image

        embedding = self._embedder.encode_image(crop)
        embedding = ClipEmbeddingService.normalize(embedding)

        logger.info(
            "Pipeline completado | detectado=%s clase=%s dim=%s",
            detection.detected,
            detection.class_name,
            embedding.shape[0],
        )
        return detection, embedding
