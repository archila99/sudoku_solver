# Sudoku Solver

A stateless full-stack Sudoku web app: manual grid input, backtracking solver with difficulty detection, and photo upload with computer-vision OCR.

| Layer | Stack | Deploy target |
|-------|-------|---------------|
| Frontend | React 19, TypeScript, Vite, Tailwind CSS | [Vercel](https://vercel.com) |
| Backend | Python 3.12, FastAPI, OpenCV, Tesseract | [Render](https://render.com) (Docker) |

No database. Upload an image → get a 9×9 board → solve in the browser.

## Features

- Interactive 9×9 grid with keyboard navigation
- Backtracking solver with difficulty labels (easy / medium / hard)
- Image upload: grid detection, perspective correction, per-cell OCR
- OCR tuned for flat screenshots, perspective photos, and colorful/noisy backgrounds
- Optional debug mode that saves every pipeline stage to disk

## How the OCR pipeline works

```
Image upload
  → Grayscale + grid enhancement (suppress colorful backgrounds)
  → Stage 2: Grid detection
       FAST path:  Hough lines → cluster to 10×10 → corner reconstruction
       ROBUST path: contour quadrilateral fallback
  → Perspective warp to 450×450
  → Validation gate (spacing, aspect ratio, edge density)
       FAST fails validation → retry with ROBUST corners
  → Stage 3: Refine 10 horizontal + 10 vertical cell boundaries
  → Stage 4: Per-cell preprocessing + Tesseract voting → 9×9 board
```

Empty or uncertain cells are returned as `0`. The solver treats `0` as blank.

---

## Quick start (local)

**Prerequisites:** Node.js 18+, Python 3.12+, [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

```bash
# macOS
brew install tesseract

# Ubuntu / Debian
sudo apt install tesseract-ocr
```

**Terminal 1 — backend**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # optional
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

**Terminal 2 — frontend**

```bash
cd frontend
npm install
cp .env.example .env.local         # optional — leave VITE_API_URL empty to use Vite proxy
npm run dev
```

Open http://localhost:5173

When `VITE_API_URL` is unset locally, Vite proxies `/solve` and `/upload` to `http://127.0.0.1:8000`.

---

## Environment variables

### Backend (`backend/.env`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CORS_ORIGINS` | Production | `http://localhost:5173` | Comma-separated allowed frontend origins |
| `TESSERACT_CMD` | No | auto-detect | Path to Tesseract binary |
| `OCR_DEBUG` | No | `false` | Save debug images to `backend/debug/` |
| `MAX_UPLOAD_BYTES` | No | `10485760` | Max upload size (10 MB) |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `PORT` | Render only | `8000` | Set automatically by Render |

### Frontend (`frontend/.env.local`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VITE_API_URL` | Production | — | Backend URL, e.g. `https://sudoku-api.onrender.com` |

---

## Deploy to GitHub + production

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/sudoku_solver.git
git push -u origin main
```

> `backend/debug/`, `backend/diagnostics/`, and `backend/test_images/` are gitignored. Add your own test images locally for OCR development.

### 2. Deploy backend to Render

Render uses Docker so Tesseract and OpenCV system libraries are installed at build time.

#### Option A — Blueprint (`render.yaml`)

1. In [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**.
2. Connect the repo. Render reads `render.yaml` at the repo root.
3. Set `CORS_ORIGINS` to your Vercel URL (e.g. `https://your-app.vercel.app`).
4. Deploy.

#### Option B — Manual Web Service

1. **New** → **Web Service** → connect repo.
2. Settings:
   - **Root Directory:** `backend`
   - **Runtime:** Docker
   - **Dockerfile Path:** `Dockerfile`
3. Environment variables:
   ```
   CORS_ORIGINS=https://your-app.vercel.app
   TESSERACT_CMD=/usr/bin/tesseract
   OCR_DEBUG=false
   LOG_LEVEL=INFO
   ```
4. **Health Check Path:** `/health`
5. Deploy.

Tesseract is installed in the Docker image:

```dockerfile
RUN apt-get install -y tesseract-ocr
```

Verify:

```bash
curl https://your-app.onrender.com/health
# {"status":"ok"}
```

### 3. Deploy frontend to Vercel

1. In [Vercel Dashboard](https://vercel.com) → **Add New Project** → import repo.
2. Settings:
   - **Framework Preset:** Vite
   - **Root Directory:** `frontend`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
3. Environment variable:
   ```
   VITE_API_URL=https://your-app.onrender.com
   ```
   No trailing slash.
4. Deploy.

`frontend/vercel.json` configures SPA routing (all routes → `index.html`).

### 4. Update CORS

Copy your Vercel URL and set it on Render:

```
CORS_ORIGINS=https://your-app.vercel.app
```

For multiple preview URLs, comma-separate origins (Render does not support wildcards).

---

## API

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /solve`

Request:

```json
{ "board": [[...9x9...]] }
```

Response:

```json
{ "solution": [[...]], "difficulty": "easy|medium|hard" }
```

### `POST /upload`

Multipart form field: `file` (JPEG, PNG, WebP, BMP, GIF, or TIFF).

Response:

```json
{ "board": [[...9x9...]] }
```

Cells that are empty or ambiguous after OCR are returned as `0`.

Example:

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@puzzle.jpg"
```

---

## Difficulty detection

| Level | Empty cells |
|-------|-------------|
| Easy | fewer than 30 |
| Medium | 30–39 |
| Hard | 40+ |

Based on the number of zeros in the puzzle before solving.

---

## Project structure

```
sudoku_solver/
├── backend/
│   ├── main.py              # FastAPI app (/health, /solve, /upload)
│   ├── config.py            # Environment settings
│   ├── solver.py            # Backtracking solver + difficulty
│   ├── ocr.py               # Grid detection, warp, cell OCR
│   ├── Dockerfile           # Production image (Tesseract + OpenCV)
│   ├── requirements.txt
│   ├── .env.example
│   └── diagnostics/         # Local OCR investigation scripts (gitignored output)
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api.ts           # API client (uses VITE_API_URL)
│   │   └── components/
│   ├── vercel.json
│   └── .env.example
├── render.yaml              # Render blueprint
└── README.md
```

---

## OCR debug mode

Set `OCR_DEBUG=true` in `backend/.env`. Each upload creates a timestamped folder under `backend/debug/`:

| File | Stage |
|------|--------|
| `01_original.png` | Decoded upload |
| `02_grayscale.png` | Grayscale conversion |
| `03_grid_enhanced.png` | Background suppression + line emphasis |
| `04_edges.png` | Canny edges for line detection |
| `04_candidate_contours.png` | Grid detection overlay |
| `04_detection.json` | Corners, path (fast/robust), metrics |
| `04_grid_structure.json` | Line counts, spacing, bounding box |
| `05_selected_contour.png` | Selected grid (green) |
| `05_robust_fallback_contour.png` | Robust retry overlay (if validation failed on fast) |
| `06_warped.png` | Perspective-corrected 450×450 board |
| `07_threshold_board.png` | Thresholded warped grid |
| `08_grid_validation.json` | Structure validation result |
| `08_grid_lines.json` | Final horizontal/vertical line positions |
| `09_cell_preprocessing.json` | Per-cell metrics and OCR results |
| `cells/raw/r*_c*.png` | All 81 extracted cells |
| `cells/processed/r*_c*.png` | All 81 OCR-ready cells |
| `ocr_report.json` | Summary of all cell reports |

Use this to see whether failure is in detection, warp, validation, segmentation, or OCR.

---

## Troubleshooting

### CORS errors in browser

- Confirm `CORS_ORIGINS` on Render exactly matches your Vercel URL (`https://`, no trailing slash).
- Redeploy the backend after changing `CORS_ORIGINS`.

### `VITE_API_URL is not configured`

Set `VITE_API_URL` in Vercel environment variables and redeploy.

### Upload returns 503 — OCR unavailable

Tesseract is not installed or not found. On Render, deploy with Docker (not native Python). Set `TESSERACT_CMD=/usr/bin/tesseract`.

### Upload returns 400 — Could not detect Sudoku grid

- Use a photo where the full 9×9 grid is visible.
- Avoid heavy crop, extreme glare, or very low contrast.
- Enable `OCR_DEBUG=true` and inspect the latest folder under `backend/debug/`.

### Upload returns 400 — No stable Sudoku grid detected

The validation gate rejected the warped grid (bad spacing, aspect ratio, or edge density). Enable `OCR_DEBUG=true` and check `08_grid_validation.json`. The pipeline may have retried with the robust contour path before failing.

### OCR reads wrong digits but grid looks correct

OCR quality depends on font, print quality, and lighting. Check `cells/processed/` in debug output. The grid detection may be correct while individual cell recognition fails — that is a Stage 4 (OCR) issue, not grid detection.

### Render cold starts

Free-tier services spin down after inactivity. The first request may take 30–60 seconds.

### Local upload works, production does not

1. Check `/health` on Render.
2. Test upload: `curl -F "file=@puzzle.jpg" https://your-app.onrender.com/upload`
3. Check Render logs for Tesseract or OpenCV errors.

---

## Local OCR development

Test images are not committed (see `.gitignore`). Add sample puzzles to `backend/test_images/` for local testing:

```bash
# From backend/ with venv active
python -c "
from pathlib import Path
from ocr import extract_board
board = extract_board(Path('test_images/your_puzzle.jpg').read_bytes())
print(sum(1 for r in board for c in r if c), 'digits detected')
"
```

Diagnostic scripts live in `backend/diagnostics/` for stage-by-stage investigation.

---

## License

This project is provided as-is for personal and educational use. Add a `LICENSE` file if you plan to open-source under a specific terms.
