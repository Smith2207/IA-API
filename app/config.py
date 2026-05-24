from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(..., alias="DATABASE_URL")
    cors_origins: str = Field(
        default="http://localhost:3000",
        alias="CORS_ORIGINS",
    )
    yolo_model: str = Field(default="yolov8n.pt", alias="YOLO_MODEL")
    clip_model: str = Field(default="ViT-B-32", alias="CLIP_MODEL")
    clip_pretrained: str = Field(default="openai", alias="CLIP_PRETRAINED")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    api_key: str | None = Field(default=None, alias="API_KEY")
    search_top_k: int = Field(default=5, alias="SEARCH_TOP_K", ge=1, le=20)
    similarity_min_percent: int = Field(
        default=0, alias="SIMILARITY_MIN_PERCENT", ge=0, le=100
    )
    http_timeout_seconds: float = Field(default=30.0, alias="HTTP_TIMEOUT_SECONDS")
    max_image_bytes: int = Field(default=10 * 1024 * 1024, alias="MAX_IMAGE_BYTES")

    @property
    def cors_origin_list(self) -> List[str]:
        raw = self.cors_origins.strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
