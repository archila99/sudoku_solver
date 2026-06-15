import type { Board, SolveResponse, UploadResponse } from "./types";

function getApiBase(): string {
  const url = import.meta.env.VITE_API_URL;
  if (import.meta.env.PROD && !url) {
    throw new Error(
      "VITE_API_URL is not configured. Set it in your Vercel environment variables.",
    );
  }
  return (url ?? "").replace(/\/$/, "");
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const message =
      body && typeof body.detail === "string"
        ? body.detail
        : `Request failed (${response.status})`;
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export async function solveSudoku(board: Board): Promise<SolveResponse> {
  const response = await fetch(`${getApiBase()}/solve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ board }),
  });
  return handleResponse<SolveResponse>(response);
}

export async function uploadImage(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${getApiBase()}/upload`, {
    method: "POST",
    body: formData,
  });
  return handleResponse<UploadResponse>(response);
}
