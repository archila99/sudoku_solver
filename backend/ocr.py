import logging
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

GRID_SIZE = 450
CELL_SIZE = GRID_SIZE // 9
DEBUG_DIR = Path(__file__).parent / "debug"
TESSERACT_CONFIGS = [
    r"--oem 3 --psm 10 -c tessedit_char_whitelist=123456789",
    r"--oem 3 --psm 8 -c tessedit_char_whitelist=123456789",
    r"--oem 3 --psm 6 -c tessedit_char_whitelist=123456789",
    r"--oem 3 --psm 13 -c tessedit_char_whitelist=123456789",
]


def extract_board(image_bytes: bytes) -> list[list[int]]:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = _find_best_grid_corners(gray)
    if corners is None:
        raise ValueError("Could not detect Sudoku grid in image")

    warped = _warp_perspective(gray, corners)
    warped = cv2.fastNlMeansDenoising(warped, None, 10, 7, 21)

    board, sample_cells = _ocr_cells(warped)
    _save_debug(warped, sample_cells)

    return board


def _find_best_grid_corners(gray: np.ndarray) -> np.ndarray | None:
    candidates: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()

    for quad in _all_quadrilaterals(gray):
        key = tuple(np.round(quad, -1).astype(int).flatten())
        if key in seen:
            continue
        seen.add(key)
        candidates.append(quad)

    line_corners = _corners_from_lines(gray)
    if line_corners is not None:
        key = tuple(np.round(line_corners, -1).astype(int).flatten())
        if key not in seen:
            candidates.append(line_corners)

    if not candidates:
        return None

    best_corners = candidates[0]
    best_score = -1.0
    for corners in candidates:
        score = _score_warp(gray, corners)
        if score > best_score:
            best_score = score
            best_corners = corners

    return best_corners


def _all_quadrilaterals(gray: np.ndarray) -> list[np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    quads: list[np.ndarray] = []

    grad_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)
    gradient = cv2.convertScaleAbs(cv2.subtract(grad_x, grad_y))
    sources = [gradient]

    for use_adaptive in (False, True):
        if use_adaptive:
            thresh = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2,
            )
        else:
            _, thresh = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        sources.extend([thresh, 255 - thresh])

    for source in sources:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(source, cv2.MORPH_CLOSE, kernel, iterations=2)
        quads.extend(_quadrilaterals_from_binary(processed))

    edges = cv2.Canny(blurred, 50, 150)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    quads.extend(_quadrilaterals_from_binary(dilated))

    return quads


def _quadrilaterals_from_binary(binary: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    image_area = binary.shape[0] * binary.shape[1]
    quads: list[np.ndarray] = []

    for contour in contours[:12]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.04:
            break

        peri = cv2.arcLength(contour, True)
        for epsilon_factor in (0.015, 0.02, 0.03, 0.04, 0.05):
            approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
            if len(approx) != 4:
                continue

            points = approx.reshape(4, 2).astype(np.float32)
            _, _, w, h = cv2.boundingRect(approx)
            aspect = w / h if h else 0
            if 0.65 < aspect < 1.5 and w > 80 and h > 80:
                quads.append(points)
            break

    return quads


def _score_warp(gray: np.ndarray, corners: np.ndarray) -> float:
    warped = _warp_perspective(gray, corners)
    binary = cv2.adaptiveThreshold(
        warped,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )
    h_peaks = _count_peaks(np.sum(binary, axis=1), min_distance=CELL_SIZE // 2)
    v_peaks = _count_peaks(np.sum(binary, axis=0), min_distance=CELL_SIZE // 2)
    return h_peaks + v_peaks


def _count_peaks(projection: np.ndarray, min_distance: int) -> int:
    if projection.size == 0:
        return 0
    threshold = np.max(projection) * 0.35
    peaks = 0
    last_peak = -min_distance
    for i, value in enumerate(projection):
        if value > threshold and i - last_peak >= min_distance:
            peaks += 1
            last_peak = i
    return peaks


def _corners_from_lines(gray: np.ndarray) -> np.ndarray | None:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=gray.shape[1] // 8,
        maxLineGap=20,
    )
    if lines is None:
        return None

    horizontals: list[float] = []
    verticals: list[float] = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 40:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if angle < 20 or angle > 160:
            horizontals.append((y1 + y2) / 2.0)
        elif 70 < angle < 110:
            verticals.append((x1 + x2) / 2.0)

    if len(horizontals) < 2 or len(verticals) < 2:
        return None

    top = min(horizontals)
    bottom = max(horizontals)
    left = min(verticals)
    right = max(verticals)

    width = right - left
    height = bottom - top
    if width < 80 or height < 80:
        return None
    aspect = width / height if height else 0
    if not (0.65 < aspect < 1.5):
        return None

    return np.array(
        [[left, top], [right, top], [right, bottom], [left, bottom]],
        dtype=np.float32,
    )


def _order_corners(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1).reshape(-1)

    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def _warp_perspective(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    ordered = _order_corners(corners)
    destination = np.array(
        [
            [0, 0],
            [GRID_SIZE - 1, 0],
            [GRID_SIZE - 1, GRID_SIZE - 1],
            [0, GRID_SIZE - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(gray, matrix, (GRID_SIZE, GRID_SIZE))


def _extract_cell(warped: np.ndarray, row: int, col: int) -> np.ndarray:
    margin = max(4, CELL_SIZE // 5)
    y1 = row * CELL_SIZE + margin
    y2 = (row + 1) * CELL_SIZE - margin
    x1 = col * CELL_SIZE + margin
    x2 = (col + 1) * CELL_SIZE - margin
    return warped[y1:y2, x1:x2]


def _ocr_cells(warped: np.ndarray) -> tuple[list[list[int]], dict[str, np.ndarray]]:
    board: list[list[int]] = []
    sample_cells: dict[str, np.ndarray] = {}
    sample_positions = {(0, 0), (0, 4), (4, 4), (8, 8), (2, 3), (5, 7)}

    for row in range(9):
        row_values: list[int] = []
        for col in range(9):
            cell = _extract_cell(warped, row, col)
            digit, processed = _recognize_cell(cell)
            row_values.append(digit)

            if (row, col) in sample_positions:
                sample_cells[f"cell_{row}_{col}_raw"] = cell
                sample_cells[f"cell_{row}_{col}_processed"] = processed

        board.append(row_values)

    return board, sample_cells


def _recognize_cell(cell: np.ndarray) -> tuple[int, np.ndarray]:
    if cell.size == 0:
        blank = _blank_cell_image()
        return 0, blank

    preprocessed_images = _preprocess_variants(cell)
    votes: dict[int, int] = {}
    best_digit = 0
    best_conf = -1
    best_image = preprocessed_images[0]

    for processed in preprocessed_images:
        if _is_empty_cell(processed):
            continue
        pil_image = Image.fromarray(processed)
        for config in TESSERACT_CONFIGS:
            digit, conf = _tesseract_read_with_confidence(pil_image, config)
            if digit == 0:
                continue
            votes[digit] = votes.get(digit, 0) + max(conf, 1)
            if conf > best_conf:
                best_conf = conf
                best_digit = digit
                best_image = processed

    if not votes:
        return 0, preprocessed_images[0]

    voted_digit = max(votes, key=lambda d: votes[d])
    if votes[voted_digit] >= votes.get(best_digit, 0):
        return voted_digit, votes[voted_digit]
    return best_digit, best_conf


def _preprocess_variants(cell: np.ndarray) -> list[np.ndarray]:
    resized = cv2.resize(cell, (100, 100), interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    variants: list[np.ndarray] = []

    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(_normalize_binary(otsu))

    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    variants.append(_normalize_binary(adaptive))

    return [_finalize_cell(binary) for binary in variants]


def _normalize_binary(binary: np.ndarray) -> np.ndarray:
    if np.mean(binary) < 127:
        binary = 255 - binary
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)


def _finalize_cell(binary: np.ndarray) -> np.ndarray:
    if _is_empty_cell(binary):
        return _blank_cell_image()
    cropped = _crop_to_digit(binary)
    return _pad_for_tesseract(cropped)


def _is_empty_cell(binary: np.ndarray) -> bool:
    ink_pixels = np.count_nonzero(binary == 0)
    if ink_pixels / binary.size < 0.012:
        return True

    inverted = 255 - binary
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return True

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 70:
        return True

    _, _, w, h = cv2.boundingRect(largest)
    return h < 6 or w < 3


def _crop_to_digit(binary: np.ndarray) -> np.ndarray:
    inverted = 255 - binary
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    pad = 5
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(binary.shape[1], x + w + pad)
    y2 = min(binary.shape[0], y + h + pad)
    cropped = binary[y1:y2, x1:x2]
    return cropped if cropped.size else binary


def _pad_for_tesseract(binary: np.ndarray) -> np.ndarray:
    target = 72
    h, w = binary.shape
    scale = target / max(h, w, 1)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(
        resized, 28, 28, 28, 28, cv2.BORDER_CONSTANT, value=255
    )


def _blank_cell_image() -> np.ndarray:
    return np.full((128, 128), 255, dtype=np.uint8)


def _tesseract_read_with_confidence(
    pil_image: Image.Image, config: str
) -> tuple[int, int]:
    data = pytesseract.image_to_data(
        pil_image, config=config, output_type=pytesseract.Output.DICT
    )

    best_digit = 0
    best_conf = -1
    for text, conf_str in zip(data["text"], data["conf"], strict=False):
        text = text.strip()
        if text not in "123456789":
            continue
        try:
            conf = int(conf_str)
        except ValueError:
            continue
        if conf > best_conf:
            best_conf = conf
            best_digit = int(text)

    if best_digit != 0 and best_conf >= 10:
        return best_digit, best_conf

    text = pytesseract.image_to_string(pil_image, config=config).strip()
    digits = [ch for ch in text if ch in "123456789"]
    if len(digits) == 1:
        return int(digits[0]), 50

    return 0, best_conf


def _save_debug(warped: np.ndarray, sample_cells: dict[str, np.ndarray]) -> None:
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(DEBUG_DIR / "warped_grid.png"), warped)
        cv2.imwrite(str(DEBUG_DIR / "warped_grid_lines.png"), _draw_grid_lines(warped))
        for name, image in sample_cells.items():
            cv2.imwrite(str(DEBUG_DIR / f"{name}.png"), image)
        logger.info("Saved OCR debug images to %s", DEBUG_DIR)
    except OSError as exc:
        logger.warning("Could not save debug images: %s", exc)


def _draw_grid_lines(warped: np.ndarray) -> np.ndarray:
    preview = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    for i in range(10):
        thickness = 2 if i % 3 == 0 else 1
        pos = i * CELL_SIZE
        cv2.line(preview, (pos, 0), (pos, GRID_SIZE), (0, 0, 255), thickness)
        cv2.line(preview, (0, pos), (GRID_SIZE, pos), (0, 0, 255), thickness)
    return preview
