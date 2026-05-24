from io import BytesIO

import httpx
from PIL import Image

from app.config import get_settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class ImageLoadError(Exception):
    """No se pudo cargar o validar la imagen."""


def _open_rgb(data: bytes) -> Image.Image:
    settings = get_settings()
    if len(data) > settings.max_image_bytes:
        raise ImageLoadError(
            f"Imagen demasiado grande (máx. {settings.max_image_bytes // (1024 * 1024)} MB)"
        )
    try:
        img = Image.open(BytesIO(data))
        return img.convert("RGB")
    except Exception as exc:
        raise ImageLoadError("Formato de imagen no válido") from exc


async def load_image_from_url(url: str) -> Image.Image:
    settings = get_settings()
    logger.info("Descargando imagen desde URL")
    try:
        async with httpx.AsyncClient(
            timeout=settings.http_timeout_seconds,
            follow_redirects=True,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if content_type and not content_type.startswith("image/"):
                raise ImageLoadError(
                    f"URL no devuelve una imagen (content-type: {content_type})"
                )
            return _open_rgb(response.content)
    except ImageLoadError:
        raise
    except httpx.HTTPError as exc:
        raise ImageLoadError(f"No se pudo descargar la imagen: {exc}") from exc


def load_image_from_bytes(data: bytes) -> Image.Image:
    return _open_rgb(data)
