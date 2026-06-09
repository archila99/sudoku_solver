import cv2
import numpy as np
import pytesseract
from PIL import Image


def extract_board(image_bytes: bytes) -> list[list[int]]:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    grid = _find_largest_grid(thresh)
    if grid is None:
        grid = _find_largest_grid(255 - thresh)
    if grid is None:
        raise ValueError("Could not detect Sudoku grid in image")

    x, y, w, h = grid
    cropped = gray[y : y + h, x : x + w]
    return _ocr_cells(cropped)


def _find_largest_grid(thresh: np.ndarray) -> tuple[int, int, int, int] | None:
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        area = w * h
        aspect = w / h if h else 0
        if area > best_area and 0.8 < aspect < 1.2 and w > 100 and h > 100:
            best_area = area
            best = (x, y, w, h)

    return best


def _ocr_cells(grid_image: np.ndarray) -> list[list[int]]:
    board: list[list[int]] = []
    height, width = grid_image.shape
    cell_h = height // 9
    cell_w = width // 9

    for row in range(9):
        row_values: list[int] = []
        for col in range(9):
            y1 = row * cell_h + cell_h // 10
            y2 = (row + 1) * cell_h - cell_h // 10
            x1 = col * cell_w + cell_w // 10
            x2 = (col + 1) * cell_w - cell_w // 10
            cell = grid_image[y1:y2, x1:x2]
            row_values.append(_read_digit(cell))
        board.append(row_values)

    return board


def _read_digit(cell: np.ndarray) -> int:
    if cell.size == 0:
        return 0

    _, binary = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = 255 - binary

    padded = cv2.copyMakeBorder(binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    pil_image = Image.fromarray(padded)

    config = r"--psm 10 -c tessedit_char_whitelist=123456789"
    text = pytesseract.image_to_string(pil_image, config=config).strip()

    if len(text) == 1 and text.isdigit():
        return int(text)
    return 0
