import json
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile
from pydantic import ValidationError

from app.exceptions import PetNotDetectedError
from app.api.schemas import (
    HealthResponse,
    RegisterJsonRequest,
    RegisterResponse,
    SearchResultItem,
)
from app.config import get_settings
from app.db.repository import PetEmbeddingRepository
from app.services.pipeline import PetVisionPipeline
from app.services.registration import PetRegistrationService
from app.services.search import FaissSearchIndex
from app.utils.image_loader import ImageLoadError, load_image_from_bytes, load_image_from_url
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    settings = get_settings()
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="API key inválida")


class SearchJsonRequest:
    def __init__(self, image_url: str, exclude_pet_id: Optional[str] = None):
        self.image_url = image_url
        self.exclude_pet_id = exclude_pet_id


async def _parse_search_json(request: Request) -> SearchJsonRequest:
    from pydantic import BaseModel, HttpUrl

    class _Body(BaseModel):
        image_url: HttpUrl
        exclude_pet_id: Optional[str] = None

    payload = _Body.model_validate(await request.json())
    return SearchJsonRequest(
        image_url=str(payload.image_url),
        exclude_pet_id=payload.exclude_pet_id,
    )


async def _load_image_from_request(
    request: Request,
    *,
    image_url: Optional[str] = None,
    file: Optional[UploadFile] = None,
):
    if file is not None:
        data = await file.read()
        return load_image_from_bytes(data)
    if image_url:
        return await load_image_from_url(image_url)
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        url = body.get("image_url")
        if url:
            return await load_image_from_url(str(url))
    raise HTTPException(
        status_code=400,
        detail="Debes enviar image_url (JSON) o un archivo image (multipart).",
    )


@router.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    repo = PetEmbeddingRepository()
    index = FaissSearchIndex.get_instance()
    return HealthResponse(
        registered_pets=repo.count(),
        faiss_vectors=index.vector_count,
    )


@router.post(
    "/register",
    response_model=RegisterResponse,
    dependencies=[Depends(verify_api_key)],
)
async def register_pet(request: Request) -> RegisterResponse:
    """
    Registra una mascota: YOLO + CLIP + Neon + FAISS.
    JSON: { pet_id, pet_name, location?, image_url }
    Multipart: pet_id, pet_name, location?, image_url?, image (archivo)
    """
    service = PetRegistrationService()
    content_type = request.headers.get("content-type", "")

    try:
        if "application/json" in content_type:
            body = RegisterJsonRequest.model_validate(await request.json())
            img = await load_image_from_url(str(body.image_url))
            result = service.register(
                pet_id=body.pet_id,
                pet_name=body.pet_name,
                location=body.location,
                image_url=str(body.image_url),
                image=img,
            )
        elif "multipart/form-data" in content_type:
            form = await request.form()
            pet_id = form.get("pet_id")
            pet_name = form.get("pet_name")
            if not pet_id or not pet_name:
                raise HTTPException(
                    status_code=400,
                    detail="pet_id y pet_name son obligatorios.",
                )
            location = form.get("location")
            image_url = form.get("image_url")
            upload = form.get("image")
            file = upload if isinstance(upload, UploadFile) else None

            if not image_url and file is None:
                raise HTTPException(
                    status_code=400,
                    detail="Debes enviar image_url o archivo image.",
                )

            img = await _load_image_from_request(
                request,
                image_url=str(image_url) if image_url else None,
                file=file,
            )
            resolved_url = str(image_url) if image_url else f"upload://{pet_id}"
            result = service.register(
                pet_id=str(pet_id),
                pet_name=str(pet_name),
                location=str(location) if location else None,
                image_url=resolved_url,
                image=img,
            )
        else:
            raise HTTPException(
                status_code=415,
                detail="Usa application/json o multipart/form-data.",
            )

        return RegisterResponse(
            pet_id=result["pet_id"],
            pet_name=result["pet_name"],
            detected_class=result.get("detected_class"),
            confidence=result.get("confidence"),
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc
    except PetNotDetectedError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ImageLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error en /register")
        raise HTTPException(
            status_code=500, detail="Error interno al registrar mascota."
        ) from exc


@router.post(
    "/search",
    response_model=List[SearchResultItem],
    dependencies=[Depends(verify_api_key)],
)
async def search_pets(request: Request) -> List[SearchResultItem]:
    """
    Compara una imagen con el índice y devuelve hasta 5 coincidencias por similitud.
    """
    pipeline = PetVisionPipeline()
    index = FaissSearchIndex.get_instance()
    settings = get_settings()
    content_type = request.headers.get("content-type", "")

    try:
        exclude: Optional[str] = None

        if "application/json" in content_type:
            search_body = await _parse_search_json(request)
            img = await load_image_from_url(search_body.image_url)
            exclude = search_body.exclude_pet_id
        elif "multipart/form-data" in content_type:
            form = await request.form()
            image_url = form.get("image_url")
            exclude = form.get("exclude_pet_id")
            if exclude is not None:
                exclude = str(exclude)
            upload = form.get("image")
            file = upload if isinstance(upload, UploadFile) else None
            img = await _load_image_from_request(
                request,
                image_url=str(image_url) if image_url else None,
                file=file,
            )
        else:
            raise HTTPException(
                status_code=415,
                detail="Usa application/json o multipart/form-data.",
            )

        detection, embedding = pipeline.process_image(img, require_pet=True)
        logger.info(
            "Búsqueda | clase=%s conf=%.1f%%",
            detection.class_name,
            detection.confidence * 100,
        )

        matches = index.search(
            embedding,
            top_k=settings.search_top_k,
            exclude_id=exclude,
        )

        return [
            SearchResultItem(
                pet_name=meta.get("pet_name", ""),
                similarity=int(round(score)),
                location=meta.get("location") or "",
                image_url=meta.get("image_url") or "",
            )
            for _pet_id, score, meta in matches
        ]
    except PetNotDetectedError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ImageLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error en /search")
        raise HTTPException(status_code=500, detail="Error interno en búsqueda.") from exc
