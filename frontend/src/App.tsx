import { useRef, useState } from "react";
import { solveSudoku, uploadImage } from "./api";
import SudokuGrid, { createEmptyBoard } from "./components/SudokuGrid";
import type { Board, Difficulty } from "./types";

const SAMPLE_PUZZLE: Board = [
  [5, 3, 0, 0, 7, 0, 0, 0, 0],
  [6, 0, 0, 1, 9, 5, 0, 0, 0],
  [0, 9, 8, 0, 0, 0, 0, 6, 0],
  [8, 0, 0, 0, 6, 0, 0, 0, 3],
  [4, 0, 0, 8, 0, 3, 0, 0, 1],
  [7, 0, 0, 0, 2, 0, 0, 0, 6],
  [0, 6, 0, 0, 0, 0, 2, 8, 0],
  [0, 0, 0, 4, 1, 9, 0, 0, 5],
  [0, 0, 0, 0, 8, 0, 0, 7, 9],
];

const difficultyColors: Record<Difficulty, string> = {
  easy: "bg-green-100 text-green-800",
  medium: "bg-yellow-100 text-yellow-800",
  hard: "bg-red-100 text-red-800",
};

export default function App() {
  const [board, setBoard] = useState<Board>(createEmptyBoard());
  const [difficulty, setDifficulty] = useState<Difficulty | null>(null);
  const [solved, setSolved] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSolve = async () => {
    setLoading(true);
    setError(null);
    setDifficulty(null);
    setSolved(false);
    try {
      const result = await solveSudoku(board);
      setBoard(result.solution);
      setDifficulty(result.difficulty);
      setSolved(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to solve puzzle");
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setBoard(createEmptyBoard());
    setDifficulty(null);
    setSolved(false);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleUpload = async (file: File) => {
    setLoading(true);
    setError(null);
    setDifficulty(null);
    setSolved(false);
    try {
      const result = await uploadImage(file);
      setBoard(result.board);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to process image");
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      void handleUpload(file);
    }
  };

  const loadSample = () => {
    setBoard(SAMPLE_PUZZLE.map((row) => [...row]));
    setDifficulty(null);
    setSolved(false);
    setError(null);
  };

  return (
    <div className="min-h-screen px-4 py-8">
      <div className="mx-auto flex max-w-lg flex-col items-center gap-6">
        <header className="text-center">
          <h1 className="text-3xl font-bold tracking-tight text-slate-900">
            Sudoku Solver
          </h1>
          <p className="mt-1 text-sm text-slate-500">
            Enter a puzzle manually or upload an image
          </p>
        </header>

        <SudokuGrid
          board={board}
          onChange={(next) => {
            setBoard(next);
            setSolved(false);
            setDifficulty(null);
          }}
          solved={solved}
          readOnly={solved}
        />

        {difficulty && (
          <div
            className={`rounded-full px-4 py-1 text-sm font-medium capitalize ${difficultyColors[difficulty]}`}
          >
            Difficulty: {difficulty}
          </div>
        )}

        {error && (
          <div className="w-full rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        {loading && (
          <div className="flex items-center gap-2 text-sm text-slate-600">
            <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-blue-600" />
            Processing…
          </div>
        )}

        <div className="flex w-full flex-wrap justify-center gap-3">
          <button
            onClick={() => void handleSolve()}
            disabled={loading}
            className="rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-medium text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Solve Puzzle
          </button>
          <button
            onClick={handleClear}
            disabled={loading}
            className="rounded-lg border border-slate-300 bg-white px-5 py-2.5 text-sm font-medium text-slate-700 shadow-sm transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Clear Board
          </button>
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={loading}
            className="rounded-lg border border-slate-300 bg-white px-5 py-2.5 text-sm font-medium text-slate-700 shadow-sm transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Upload Image
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFileChange}
          />
        </div>

        <button
          onClick={loadSample}
          disabled={loading}
          className="text-sm text-blue-600 underline-offset-2 hover:underline disabled:opacity-50"
        >
          Load sample puzzle
        </button>
      </div>
    </div>
  );
}
