import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from config import get_settings
from ocr import configure_tesseract, extract_board, verify_tesseract
from solver import detect_difficulty, solve

settings = get_settings()

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_tesseract()
    if verify_tesseract():
        logger.info("Tesseract OCR is available")
    else:
        logger.warning("Tesseract OCR is not available — image upload will fail")
    logger.info("CORS origins: %s", ", ".join(settings.cors_origins))
    yield


app = FastAPI(title="Sudoku Solver API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class SolveRequest(BaseModel):
    board: list[list[int]]

    @field_validator("board")
    @classmethod
    def validate_board(cls, board: list[list[int]]) -> list[list[int]]:
        if len(board) != 9 or any(len(row) != 9 for row in board):
            raise ValueError("Board must be 9x9")
        for row in board:
            for cell in row:
                if not isinstance(cell, int) or cell < 0 or cell > 9:
                    raise ValueError("Each cell must be an integer from 0 to 9")
        return board


class SolveResponse(BaseModel):
    solution: list[list[int]]
    difficulty: str


class UploadResponse(BaseModel):
    board: list[list[int]]


class ErrorResponse(BaseModel):
    detail: str


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _: object, exc: RequestValidationError
) -> JSONResponse:
    errors = exc.errors()
    message = errors[0]["msg"] if errors else "Invalid request"
    return JSONResponse(status_code=422, content={"detail": message})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: object, exc: Exception) -> JSONResponse:
    if isinstance(exc, StarletteHTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return JSONResponse(status_code=exc.status_code, content={"detail": detail})

    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred"},
    )


def _is_image_upload(file: UploadFile) -> bool:
    if file.content_type and file.content_type.startswith("image/"):
        return True
    if file.filename:
        return Path(file.filename).suffix.lower() in IMAGE_EXTENSIONS
    return False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/solve", response_model=SolveResponse, responses={400: {"model": ErrorResponse}})
def solve_puzzle(request: SolveRequest) -> SolveResponse:
    difficulty = detect_difficulty(request.board)
    solution = solve(request.board)
    if solution is None:
        raise HTTPException(status_code=400, detail="Puzzle is invalid or unsolvable")
    return SolveResponse(solution=solution, difficulty=difficulty)


@app.post("/upload", response_model=UploadResponse, responses={400: {"model": ErrorResponse}})
async def upload_image(file: UploadFile = File(...)) -> UploadResponse:
    if not _is_image_upload(file):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {settings.max_upload_bytes // (1024 * 1024)} MB)",
        )

    if not verify_tesseract():
        raise HTTPException(
            status_code=503,
            detail="OCR service is unavailable. Tesseract is not installed on the server.",
        )

    try:
        board = extract_board(image_bytes)
    except ValueError as exc:
        logger.warning("OCR failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected OCR error")
        raise HTTPException(
            status_code=500,
            detail="Failed to process image",
        ) from exc

    return UploadResponse(board=board)
