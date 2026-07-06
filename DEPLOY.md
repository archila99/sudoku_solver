# Deployment guide — Render (backend) + Vercel (frontend)

Repo: `https://github.com/archila99/sudoku_solver`

Deploy **backend first**, then frontend, then wire CORS.

---

## Prerequisites

- GitHub repo pushed and up to date
- [Render](https://render.com) account
- [Vercel](https://vercel.com) account

---

## Step 1 — Deploy backend to Render

### Option A: Blueprint (recommended)

1. Open [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**.
2. Connect GitHub → select `archila99/sudoku_solver`.
3. Render reads `render.yaml` from the repo root.
4. When prompted, set **`CORS_ORIGINS`** — you can use a placeholder for now:
   ```
   https://placeholder.vercel.app
   ```
   You will update this after Vercel deploy (Step 3).
5. Click **Apply** / **Deploy**.

### Option B: Manual web service

| Setting | Value |
|---------|--------|
| Name | `sudoku-backend` |
| Region | closest to your users |
| Branch | `main` |
| Root Directory | `backend` |
| Runtime | **Docker** |
| Dockerfile path | `Dockerfile` |
| Instance type | Free (or paid for faster OCR) |
| Health check path | `/health` |

**Environment variables:**

| Key | Value |
|-----|--------|
| `CORS_ORIGINS` | `https://your-app.vercel.app` (update after Step 2) |
| `TESSERACT_CMD` | `/usr/bin/tesseract` |
| `OCR_DEBUG` | `false` |
| `LOG_LEVEL` | `INFO` |
| `MAX_UPLOAD_BYTES` | `10485760` |

`PORT` is set automatically by Render — do not override.

### Verify backend

After deploy finishes (first build ~3–5 min):

```bash
curl https://<your-render-service>.onrender.com/health
```

Expected:

```json
{"status":"ok"}
```

Test solve:

```bash
curl -X POST https://<your-render-service>.onrender.com/solve \
  -H "Content-Type: application/json" \
  -d '{"board":[[5,3,0,0,7,0,0,0,0],[6,0,0,1,9,5,0,0,0],[0,9,8,0,0,0,0,6,0],[8,0,0,0,6,0,0,0,3],[4,0,0,8,0,3,0,0,1],[7,0,0,0,2,0,0,0,6],[0,6,0,0,0,0,2,8,0],[0,0,0,4,1,9,0,0,5],[0,0,0,0,8,0,0,7,9]]}'
```

Copy your Render URL — you need it for Vercel:

```
https://<your-render-service>.onrender.com
```

No trailing slash.

---

## Step 2 — Deploy frontend to Vercel

1. Open [Vercel Dashboard](https://vercel.com) → **Add New** → **Project**.
2. Import `archila99/sudoku_solver` from GitHub.
3. Configure project:

| Setting | Value |
|---------|--------|
| Framework Preset | Vite (auto-detected) |
| Root Directory | `frontend` |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| Install Command | `npm install` |

4. **Environment variables** (Production):

| Key | Value |
|-----|--------|
| `VITE_API_URL` | `https://<your-render-service>.onrender.com` |

5. Deploy.

6. Copy your Vercel URL, e.g. `https://sudoku-solver.vercel.app`.

### Verify frontend

Open the Vercel URL in a browser:

- Click **Load sample puzzle** → **Solve Puzzle** — should return a solved grid.
- Try **Upload Image** with a clear Sudoku photo.

---

## Step 3 — Connect CORS (required)

In Render → your web service → **Environment**:

```
CORS_ORIGINS=https://<your-vercel-app>.vercel.app
```

Rules:

- Include `https://`
- No trailing slash
- Must match the browser origin exactly

For multiple origins (e.g. preview + production):

```
CORS_ORIGINS=https://sudoku-solver.vercel.app,https://sudoku-solver-git-main-archila99.vercel.app
```

Save → Render redeploys automatically.

---

## Step 4 — Smoke test production

```bash
# Health
curl https://<render-url>/health

# Solve
curl -X POST https://<render-url>/solve \
  -H "Content-Type: application/json" \
  -d '{"board":[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]}'

# Upload (optional)
curl -X POST https://<render-url>/upload \
  -F "file=@puzzle.jpg"
```

In the browser, open DevTools → Network. Confirm `/solve` and `/upload` hit your Render URL, not `localhost`.

---

## Architecture

```
Browser (Vercel)
  │  VITE_API_URL → Render backend
  ▼
Render Docker container
  ├── FastAPI (main.py)
  ├── Tesseract OCR
  └── OpenCV
```

---

## Troubleshooting

### CORS error in browser

- `CORS_ORIGINS` on Render must exactly match the Vercel URL.
- Redeploy backend after changing env vars.

### `VITE_API_URL is not configured`

Set `VITE_API_URL` in Vercel → Settings → Environment Variables → **Production** → redeploy.

### Upload returns 503

Tesseract missing. Confirm Render uses **Docker** runtime (not native Python). `TESSERACT_CMD=/usr/bin/tesseract` should be set.

### Upload times out on free tier

OCR can take 30–60 seconds on large images. Render free tier has request timeouts. Options:

- Use smaller/cropped puzzle photos
- Upgrade Render plan for longer timeouts

### Cold start (free tier)

First request after idle may take 30–60 seconds. `/health` warms the service.

### Build fails on Vercel

- Root Directory must be `frontend`
- Node 18+ (set in `package.json` engines)

### Build fails on Render

- Check logs for Docker/apt errors
- Confirm `backend/Dockerfile` and `backend/start.sh` are in the repo

---

## Redeploying updates

Push to `main`:

```bash
git add .
git commit -m "Your change"
git push origin main
```

- Render: auto-deploys if `autoDeploy: true` in `render.yaml`
- Vercel: auto-deploys on push to connected branch

---

## Checklist

- [ ] Backend deployed on Render (Docker)
- [ ] `GET /health` returns `{"status":"ok"}`
- [ ] Frontend deployed on Vercel with `Root Directory = frontend`
- [ ] `VITE_API_URL` set to Render URL
- [ ] `CORS_ORIGINS` set to Vercel URL on Render
- [ ] Sample puzzle solves in production UI
- [ ] Image upload tested (optional)
