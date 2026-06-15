# Sudoku Solver

A stateless full-stack Sudoku web app: manual input, puzzle solving, difficulty detection, and image OCR upload.

- **Frontend:** React, TypeScript, Vite, Tailwind CSS → [Vercel](https://vercel.com)
- **Backend:** Python, FastAPI → [Render](https://render.com)
- **OCR:** OpenCV + Tesseract (no database)

## Features

- 9×9 Sudoku grid with keyboard navigation
- Backtracking solver with difficulty detection (easy / medium / hard)
- Image upload with grid detection and per-cell OCR

---

## Local development

### Prerequisites

- Node.js 18+
- Python 3.12+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

```bash
# macOS
brew install tesseract

# Ubuntu / Debian
sudo apt install tesseract-ocr
```

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # optional
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### Frontend

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

## Deploy backend to Render

Render uses Docker so Tesseract and OpenCV system libraries are installed at build time.

### Option A — Blueprint (`render.yaml`)

1. Push this repo to GitHub.
2. In [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**.
3. Connect the repo. Render reads `render.yaml` at the repo root.
4. Set `CORS_ORIGINS` to your Vercel URL (e.g. `https://your-app.vercel.app`).
5. Deploy.

### Option B — Manual Web Service

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

### OCR on Render

Tesseract is installed in the Docker image via:

```dockerfile
RUN apt-get install -y tesseract-ocr
```

`TESSERACT_CMD=/usr/bin/tesseract` is set in the Dockerfile and `render.yaml`. No extra build steps needed.

### Verify backend

```bash
curl https://your-app.onrender.com/health
# {"status":"ok"}

curl -X POST https://your-app.onrender.com/solve \
  -H "Content-Type: application/json" \
  -d '{"board":[[5,3,0,0,7,0,0,0,0],[6,0,0,1,9,5,0,0,0],[0,9,8,0,0,0,0,6,0],[8,0,0,0,6,0,0,0,3],[4,0,0,8,0,3,0,0,1],[7,0,0,0,2,0,0,0,6],[0,6,0,0,0,0,2,8,0],[0,0,0,4,1,9,0,0,5],[0,0,0,0,8,0,0,7,9]]}'
```

---

## Deploy frontend to Vercel

1. Push this repo to GitHub.
2. In [Vercel Dashboard](https://vercel.com) → **Add New Project** → import repo.
3. Settings:
   - **Framework Preset:** Vite
   - **Root Directory:** `frontend`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
4. Environment variables:
   ```
   VITE_API_URL=https://your-app.onrender.com
   ```
   No trailing slash.
5. Deploy.

`frontend/vercel.json` configures SPA routing (all routes → `index.html`).

### Update CORS after Vercel deploy

Copy your Vercel URL and set it on Render:

```
CORS_ORIGINS=https://your-app.vercel.app
```

For preview deployments, add multiple origins comma-separated:

```
CORS_ORIGINS=https://your-app.vercel.app,https://your-app-*.vercel.app
```

(Render does not support wildcards — add each preview URL explicitly, or use a custom domain.)

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

Multipart form field: `file` (image).

Response:

```json
{ "board": [[...9x9...]] }
```

Empty or uncertain cells are returned as `0`.

---

## Difficulty

| Level | Empty cells |
|-------|-------------|
| Easy | fewer than 30 |
| Medium | 30–39 |
| Hard | 40+ |

---

## Troubleshooting

### CORS errors in browser

- Confirm `CORS_ORIGINS` on Render exactly matches your Vercel URL (including `https://`, no trailing slash).
- Redeploy the backend after changing `CORS_ORIGINS`.

### `VITE_API_URL is not configured`

Set `VITE_API_URL` in Vercel environment variables and redeploy.

### Upload returns 503 — OCR unavailable

Tesseract is not installed or not found. On Render, ensure you deploy with Docker (not native Python). Set `TESSERACT_CMD=/usr/bin/tesseract`.

### Upload returns 400 — Could not detect grid

- Use a clear photo with the full Sudoku grid visible.
- Avoid heavy skew, glare, or cropped grids.
- Enable `OCR_DEBUG=true` on the backend to inspect `backend/debug/` images.

### Render cold starts

Free-tier services spin down after inactivity. First request may take 30–60 seconds.

### Local upload works, production does not

1. Check `/health` on Render.
2. Test upload with `curl -F "file=@puzzle.jpg" https://your-app.onrender.com/upload`.
3. Check Render logs for Tesseract or OpenCV errors.

---

## Project structure

```
sudoku_solver/
├── backend/
│   ├── main.py          # FastAPI app
│   ├── config.py        # Environment settings
│   ├── solver.py        # Backtracking solver
│   ├── ocr.py           # Image processing + OCR
│   ├── Dockerfile       # Render production image
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── api.ts       # API client (uses VITE_API_URL)
│   │   └── ...
│   └── vercel.json
└── render.yaml          # Render blueprint
```
