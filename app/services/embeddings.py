from typing import Optional

import numpy as np
import open_clip
import torch
from PIL import Image

from app.config import get_settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Prompts enriquecidos para capturar rasgos visuales (pelaje, forma, raza, cara, manchas)
VISUAL_PROMPTS = [
    "a photo of a dog",
    "a photo of a cat",
    "pet fur color and coat pattern",
    "pet body shape and size",
    "pet breed appearance",
    "pet facial structure and eyes",
    "pet spots stripes or unique markings",
]


class ClipEmbeddingService:
    """Embeddings visuales con OpenAI CLIP (open_clip)."""

    _instance: Optional["ClipEmbeddingService"] = None

    def __init__(self) -> None:
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device: Optional[torch.device] = None
        self._text_features: Optional[torch.Tensor] = None

    @classmethod
    def get_instance(cls) -> "ClipEmbeddingService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self) -> None:
        if self._model is not None:
            return

        settings = get_settings()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "Cargando CLIP %s (%s) en %s",
            settings.clip_model,
            settings.clip_pretrained,
            self._device,
        )

        model, _, preprocess = open_clip.create_model_and_transforms(
            settings.clip_model,
            pretrained=settings.clip_pretrained,
        )
        tokenizer = open_clip.get_tokenizer(settings.clip_model)

        model.eval()
        model.to(self._device)

        with torch.no_grad():
            tokens = tokenizer(VISUAL_PROMPTS).to(self._device)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self._model = model
        self._preprocess = preprocess
        self._tokenizer = tokenizer
        self._text_features = text_features

    def encode_image(self, image: Image.Image) -> np.ndarray:
        self._load_model()
        assert self._model is not None
        assert self._preprocess is not None
        assert self._text_features is not None
        assert self._device is not None

        tensor = self._preprocess(image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            image_features = self._model.encode_image(tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Fusión imagen + contexto visual (promedio ponderado)
            combined = 0.75 * image_features + 0.25 * self._text_features.mean(dim=0, keepdim=True)
            combined = combined / combined.norm(dim=-1, keepdim=True)

        vector = combined.cpu().numpy().astype(np.float32).flatten()
        return vector

    @staticmethod
    def normalize(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            return vector.astype(np.float32)
        return (vector / norm).astype(np.float32)
