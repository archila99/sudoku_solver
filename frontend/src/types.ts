export type Board = number[][];
export type Difficulty = "easy" | "medium" | "hard";

export interface SolveResponse {
  solution: Board;
  difficulty: Difficulty;
}

export interface UploadResponse {
  board: Board;
}
