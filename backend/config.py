import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    cors_origins: tuple[str, ...]
    tesseract_cmd: str | None
    ocr_debug: bool
    max_upload_bytes: int
    log_level: str


@lru_cache
def get_settings() -> Settings:
    origins_raw = os.getenv("CORS_ORIGINS", "http://localhost:5173")
    origins = tuple(origin.strip() for origin in origins_raw.split(",") if origin.strip())

    return Settings(
        cors_origins=origins,
        tesseract_cmd=os.getenv("TESSERACT_CMD"),
        ocr_debug=os.getenv("OCR_DEBUG", "false").lower() in ("1", "true", "yes"),
        max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
