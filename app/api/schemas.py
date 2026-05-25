from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl


class HealthResponse(BaseModel):
    service: str = "PawPatrol IA API"
    status: str = "ok"
    registered_pets: int = 0
    faiss_vectors: int = 0


class RegisterJsonRequest(BaseModel):
    pet_id: str = Field(..., min_length=1, max_length=128)
    pet_name: str = Field(..., min_length=1, max_length=200)
    location: Optional[str] = Field(default=None, max_length=300)
    image_url: Optional[HttpUrl] = None
    image_base64: Optional[str] = Field(
        default=None,
        description="Imagen en base64 (recomendado desde PawPatroll)",
    )


class RegisterResponse(BaseModel):
    success: bool = True
    pet_id: str
    pet_name: str
    detected_class: Optional[str] = None
    confidence: Optional[float] = None
    message: str = "Mascota registrada en el índice de similitud visual"


class QueryDetectionInfo(BaseModel):
    detected_class: Optional[str] = None
    confidence: Optional[float] = None


class SearchResultItem(BaseModel):
    pet_id: str
    pet_name: str
    similarity: int
    location: str = ""
    image_url: str


class SearchResponse(BaseModel):
    detection: QueryDetectionInfo
    matches: List[SearchResultItem] = Field(default_factory=list)
