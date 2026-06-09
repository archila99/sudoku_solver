import { KeyboardEvent, useRef } from "react";
import type { Board } from "../types";

interface SudokuGridProps {
  board: Board;
  onChange: (board: Board) => void;
  solved?: boolean;
  readOnly?: boolean;
}

function createEmptyBoard(): Board {
  return Array.from({ length: 9 }, () => Array(9).fill(0));
}

export { createEmptyBoard };

function isThickRight(col: number): boolean {
  return col === 2 || col === 5;
}

function isThickBottom(row: number): boolean {
  return row === 2 || row === 5;
}

export default function SudokuGrid({
  board,
  onChange,
  solved = false,
  readOnly = false,
}: SudokuGridProps) {
  const inputRefs = useRef<(HTMLInputElement | null)[][]>(
    Array.from({ length: 9 }, () => Array(9).fill(null)),
  );

  const updateCell = (row: number, col: number, value: number) => {
    const next = board.map((r) => [...r]);
    next[row][col] = value;
    onChange(next);
  };

  const focusCell = (row: number, col: number) => {
    if (row < 0 || row > 8 || col < 0 || col > 8) return;
    inputRefs.current[row][col]?.focus();
  };

  const handleKeyDown = (
    event: KeyboardEvent<HTMLInputElement>,
    row: number,
    col: number,
  ) => {
    switch (event.key) {
      case "Backspace":
      case "Delete":
        event.preventDefault();
        updateCell(row, col, 0);
        break;
      case "ArrowUp":
        event.preventDefault();
        focusCell(row - 1, col);
        break;
      case "ArrowDown":
        event.preventDefault();
        focusCell(row + 1, col);
        break;
      case "ArrowLeft":
        event.preventDefault();
        focusCell(row, col - 1);
        break;
      case "ArrowRight":
        event.preventDefault();
        focusCell(row, col + 1);
        break;
      default:
        break;
    }
  };

  const handleChange = (row: number, col: number, raw: string) => {
    const digit = raw.replace(/\D/g, "").slice(-1);
    updateCell(row, col, digit ? parseInt(digit, 10) : 0);
    if (digit && col < 8) {
      focusCell(row, col + 1);
    }
  };

  return (
    <div
      className={`mx-auto w-full max-w-md rounded-lg border-4 border-slate-800 bg-white p-1 shadow-lg ${
        solved ? "ring-4 ring-emerald-300" : ""
      }`}
    >
      <div className="grid grid-cols-9">
        {board.map((row, rowIndex) =>
          row.map((cell, colIndex) => {
            const borderClasses = [
              "border border-slate-300",
              isThickRight(colIndex) ? "border-r-2 border-r-slate-800" : "",
              isThickBottom(rowIndex) ? "border-b-2 border-b-slate-800" : "",
              colIndex === 0 ? "border-l-2 border-l-slate-800" : "",
              rowIndex === 0 ? "border-t-2 border-t-slate-800" : "",
              colIndex === 8 ? "border-r-2 border-r-slate-800" : "",
              rowIndex === 8 ? "border-b-2 border-b-slate-800" : "",
            ].join(" ");

            return (
              <input
                key={`${rowIndex}-${colIndex}`}
                ref={(el) => {
                  inputRefs.current[rowIndex][colIndex] = el;
                }}
                type="text"
                inputMode="numeric"
                maxLength={1}
                value={cell === 0 ? "" : String(cell)}
                readOnly={readOnly}
                onChange={(e) => handleChange(rowIndex, colIndex, e.target.value)}
                onKeyDown={(e) => handleKeyDown(e, rowIndex, colIndex)}
                className={`aspect-square w-full text-center text-lg font-semibold outline-none transition-colors sm:text-xl ${borderClasses} ${
                  readOnly
                    ? "bg-emerald-50 text-emerald-800"
                    : "bg-white text-slate-800 focus:bg-blue-50 focus:ring-2 focus:ring-inset focus:ring-blue-400"
                }`}
                aria-label={`Row ${rowIndex + 1}, column ${colIndex + 1}`}
              />
            );
          }),
        )}
      </div>
    </div>
  );
}
