from copy import deepcopy

Board = list[list[int]]


def count_empty_cells(board: Board) -> int:
    return sum(1 for row in board for cell in row if cell == 0)


def detect_difficulty(board: Board) -> str:
    empty = count_empty_cells(board)
    if empty < 30:
        return "easy"
    if empty < 40:
        return "medium"
    return "hard"


def is_valid(board: Board, row: int, col: int, num: int) -> bool:
    if num in board[row]:
        return False
    if any(board[r][col] == num for r in range(9)):
        return False
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if board[r][c] == num:
                return False
    return True


def find_empty(board: Board) -> tuple[int, int] | None:
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                return r, c
    return None


def is_valid_board(board: Board) -> bool:
    for r in range(9):
        for c in range(9):
            val = board[r][c]
            if val == 0:
                continue
            if val < 1 or val > 9:
                return False
            board[r][c] = 0
            if not is_valid(board, r, c, val):
                board[r][c] = val
                return False
            board[r][c] = val
    return True


def solve(board: Board) -> Board | None:
    if not is_valid_board(deepcopy(board)):
        return None

    working = deepcopy(board)
    return _backtrack(working)


def _backtrack(board: Board) -> Board | None:
    empty = find_empty(board)
    if empty is None:
        return board

    row, col = empty
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num
            result = _backtrack(board)
            if result is not None:
                return result
            board[row][col] = 0
    return None
