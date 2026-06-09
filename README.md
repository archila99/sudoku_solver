# Sudoku Solver MVP

A minimal full-stack Sudoku web app with manual input, puzzle solving, difficulty detection, and image OCR upload.

## Stack

- **Frontend:** React, TypeScript, Vite, Tailwind CSS
- **Backend:** Python, FastAPI

## Features

- 9×9 Sudoku grid with keyboard navigation
- Backtracking solver with difficulty detection (easy / medium / hard)
- Image upload with OpenCV grid detection + Tesseract OCR

## Setup

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) on your system (required for image upload):

- **macOS:** `brew install tesseract`
- **Ubuntu:** `sudo apt install tesseract-ocr`

```bash
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## API

### `POST /solve`

```json
{ "board": [[...9x9...]] }
```

Returns `{ "solution": [[...]], "difficulty": "easy|medium|hard" }`.

### `POST /upload`

Multipart form with `file` (image). Returns `{ "board": [[...]] }`.

## Difficulty

- **Easy:** fewer than 30 empty cells
- **Medium:** 30–39 empty cells
- **Hard:** 40+ empty cells
