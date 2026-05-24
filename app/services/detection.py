from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from app.config import get_settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# COCO: 15 = cat, 16 = dog
COCO_PET_CLASSES = {15: "cat", 16: "dog"}


@dataclass
class DetectionResult:
    detected: bool
    class_name: Optional[str]
    confidence: float
    bbox: Optional[Tuple[float, float, float, float]]
    crop: Image.Image


class PetDetectionService:
    """Detección de perros y gatos con YOLOv8."""

    _instance: Optional["PetDetectionService"] = None

    def __init__(self) -> None:
        self._model = None

    @classmethod
    def get_instance(cls) -> "PetDetectionService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        if self._model is not None:
            return
        from ultralytics import YOLO

        settings = get_settings()
        logger.info("Cargando YOLOv8: %s", settings.yolo_model)
        self._model = YOLO(settings.yolo_model)

    def detect(self, image: Image.Image) -> DetectionResult:
        self._load_model()
        assert self._model is not None

        results = self._model.predict(
            source=np.array(image),
            verbose=False,
            classes=list(COCO_PET_CLASSES.keys()),
        )

        best: Optional[DetectionResult] = None

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            for box in result.boxes:
                cls_id = int(box.cls.item())
                if cls_id not in COCO_PET_CLASSES:
                    continue
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                candidate = DetectionResult(
                    detected=True,
                    class_name=COCO_PET_CLASSES[cls_id],
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    crop=image.crop((int(x1), int(y1), int(x2), int(y2))),
                )
                if best is None or candidate.confidence > best.confidence:
                    best = candidate

        if best is not None:
            logger.info(
                "Mascota detectada: %s (%.2f%%)",
                best.class_name,
                best.confidence * 100,
            )
            return best

        logger.warning("No se detectó perro ni gato en la imagen")
        return DetectionResult(
            detected=False,
            class_name=None,
            confidence=0.0,
            bbox=None,
            crop=image,
        )
