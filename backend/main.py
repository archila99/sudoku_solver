from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from ocr import extract_board
from solver import detect_difficulty, solve

app = FastAPI(title="Sudoku MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
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
                    raise ValueError("Each cell must be 0-9")
        return board


class SolveResponse(BaseModel):
    solution: list[list[int]]
    difficulty: str


class UploadResponse(BaseModel):
    board: list[list[int]]


@app.post("/solve", response_model=SolveResponse)
def solve_puzzle(request: SolveRequest) -> SolveResponse:
    difficulty = detect_difficulty(request.board)
    solution = solve(request.board)
    if solution is None:
        raise HTTPException(status_code=400, detail="Puzzle is invalid or unsolvable")
    return SolveResponse(solution=solution, difficulty=difficulty)


@app.post("/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)) -> UploadResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        board = extract_board(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return UploadResponse(board=board)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
