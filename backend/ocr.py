"""Sudoku board extraction from images using OpenCV + Tesseract."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

from config import get_settings

logger = logging.getLogger(__name__)

GRID_SIZE = 450
CELL_SIZE = GRID_SIZE // 9
DEBUG_DIR = Path(__file__).parent / "debug"
_tesseract_configured = False

TESSERACT_CONFIGS = [
    r"--oem 3 --psm 10 -c tessedit_char_whitelist=123456789",
    r"--oem 3 --psm 8 -c tessedit_char_whitelist=123456789",
    r"--oem 3 --psm 6 -c tessedit_char_whitelist=123456789",
    r"--oem 3 --psm 13 -c tessedit_char_whitelist=123456789",
]

# Confidence: reject only when votes are genuinely split.
MIN_ACCEPT_CONFIDENCE = 20
MIN_VOTE_MARGIN_RATIO = 1.35

# --- Stage 3/4: cell extraction, preprocessing & OCR ---
CELL_MARGIN_RATIO = 0.08
BORDER_TOUCH_THRESHOLD = 2
EMPTY_FOREGROUND_RATIO = 0.012
MIN_LARGEST_COMPONENT_AREA = 70
MIN_COMPONENT_AREA = 70
OPEN_KERNEL_SIZE = 0  # 0 = disabled; try 2 or 3 (2 may erode thin "1" digits)
CENTERED_SQUARE_SIZE = 32
OCR_CANVAS_SIZE = 96
BORDER_DIGIT_ACTION = "discard"  # "discard" | "lower_confidence"
BORDER_DIGIT_CONFIDENCE_PENALTY = 35
CELL_WORKING_SIZE = 100
DEBUG_SAMPLE_CELLS: frozenset[tuple[int, int]] = frozenset(
  {(0, 0), (0, 8), (8, 0), (8, 8), (4, 4), (1, 2), (3, 5), (6, 7), (2, 6)}
)


@dataclass
class CellPreprocessMetrics:
  row: int
  col: int
  coords: dict[str, int]
  margin_ratio: float
  margin_x: int
  margin_y: int
  foreground_pixels: int
  foreground_percent: float
  component_count: int
  largest_component_area: int
  bounding_box: dict[str, int] | None
  suspicious_border: bool
  empty: bool

  def to_dict(self) -> dict:
    return {
      "row": self.row,
      "col": self.col,
      "coords": self.coords,
      "margin_ratio": self.margin_ratio,
      "margin_x": self.margin_x,
      "margin_y": self.margin_y,
      "foreground_pixels": self.foreground_pixels,
      "foreground_percent": round(self.foreground_percent, 5),
      "component_count": self.component_count,
      "largest_component_area": self.largest_component_area,
      "bounding_box": self.bounding_box,
      "suspicious_border": self.suspicious_border,
      "empty": self.empty,
    }


@dataclass
class CellPreprocessResult:
  original: np.ndarray
  threshold: np.ndarray
  border_removed: np.ndarray
  centered: np.ndarray
  final: np.ndarray
  metrics: CellPreprocessMetrics
  empty: bool
  suspicious_border: bool


@dataclass
class CellOcrResult:
  digit: int
  confidence: int
  processed: np.ndarray
  votes: dict[int, int]
  empty: bool
  ambiguous: bool
  metrics: CellPreprocessMetrics | None = None


@dataclass
class DebugSession:
  """Collects per-stage artifacts when OCR_DEBUG is enabled."""

  enabled: bool
  run_dir: Path | None = None
  cell_reports: list[dict] = field(default_factory=list)

  def __post_init__(self) -> None:
    if self.enabled:
      stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
      self.run_dir = DEBUG_DIR / stamp
      self.run_dir.mkdir(parents=True, exist_ok=True)

  def save_image(self, name: str, image: np.ndarray) -> None:
    if not self.enabled or self.run_dir is None:
      return
    path = self.run_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)

  def save_json(self, name: str, data: object) -> None:
    if not self.enabled or self.run_dir is None:
      return
    path = self.run_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

  def log_cell(self, report: dict) -> None:
    if self.enabled:
      self.cell_reports.append(report)

  def finalize(self) -> None:
    if self.enabled and self.run_dir is not None:
      self.save_json("ocr_report.json", self.cell_reports)
      logger.info("OCR debug saved to %s", self.run_dir)


def configure_tesseract() -> None:
  global _tesseract_configured
  if _tesseract_configured:
    return

  settings = get_settings()
  if settings.tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    _tesseract_configured = True
    return

  for candidate in (
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    shutil.which("tesseract"),
  ):
    if candidate and Path(candidate).exists():
      pytesseract.pytesseract.tesseract_cmd = candidate
      logger.info("Using Tesseract at %s", candidate)
      _tesseract_configured = True
      return

  _tesseract_configured = True


def verify_tesseract() -> bool:
  configure_tesseract()
  try:
    pytesseract.get_tesseract_version()
    return True
  except (pytesseract.TesseractNotFoundError, OSError):
    return False


def extract_board(image_bytes: bytes) -> list[list[int]]:
  debug = DebugSession(enabled=get_settings().ocr_debug)

  image_array = np.frombuffer(image_bytes, dtype=np.uint8)
  image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
  if image is None:
    raise ValueError("Could not read image")

  debug.save_image("01_original.png", image)

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  debug.save_image("02_grayscale.png", gray)

  detection_enhanced = _enhance_grid_visibility(gray)
  debug.save_image("03_grid_enhanced.png", detection_enhanced)

  corners, candidate_debug = _find_best_grid_corners(detection_enhanced, debug)
  if corners is None:
    raise ValueError("Could not detect Sudoku grid in image")

  debug.save_image("05_selected_contour.png", candidate_debug["selected_overlay"])

  detection_path = candidate_debug.get("detection", {}).get("path", "robust")
  warped, h_lines, v_lines, corners, validation = _apply_grid_validation_gate(
    gray,
    detection_enhanced,
    corners,
    detection_path,
    debug,
  )
  debug.save_image("06_warped.png", warped)

  threshold_board = cv2.adaptiveThreshold(
    warped,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2,
  )
  debug.save_image("07_threshold_board.png", threshold_board)
  debug.save_image("07_threshold_board_lines.png", _draw_grid_lines_on(threshold_board))

  debug.save_json(
    "08_grid_lines.json",
    {"horizontal": h_lines, "vertical": v_lines, "validation": validation},
  )

  board = _ocr_cells(warped, h_lines, v_lines, debug)
  debug.finalize()
  return board


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
  clip = 2.0
  tile = max(4, min(gray.shape) // 64)
  clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
  return clahe.apply(gray)


def _enhance_grid_visibility(gray: np.ndarray) -> np.ndarray:
  """
  Pre-Stage-2 enhancement: suppress colorful/textured backgrounds and
  emphasize straight Sudoku grid lines for Hough/contour detection.
  """
  if gray.ndim == 3:
    working = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
  else:
    working = gray

  clahe = _apply_clahe(working)
  block = _odd_block_size(clahe.shape)
  binary = cv2.adaptiveThreshold(
    clahe,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    block,
    2,
  )
  binary = _normalize_binary(binary)
  structure_source = 255 - binary

  height, width = clahe.shape[:2]
  horizontal_size = max(9, width // 10)
  vertical_size = max(9, height // 10)
  horizontal_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT, (horizontal_size, 1)
  )
  vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))

  horizontal_lines = cv2.morphologyEx(
    structure_source, cv2.MORPH_OPEN, horizontal_kernel
  )
  vertical_lines = cv2.morphologyEx(
    structure_source, cv2.MORPH_OPEN, vertical_kernel
  )
  grid_structure = cv2.bitwise_or(horizontal_lines, vertical_lines)

  stitch_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  grid_structure = cv2.morphologyEx(grid_structure, cv2.MORPH_CLOSE, stitch_kernel)

  smooth_background = cv2.GaussianBlur(clahe, (21, 21), 0)
  softened = cv2.addWeighted(clahe, 0.35, smooth_background, 0.65, 0)
  enhanced = softened.copy()
  line_mask = grid_structure > 0
  enhanced[line_mask] = np.clip(
    clahe[line_mask].astype(np.float32) * 0.15, 0, 80
  ).astype(np.uint8)
  return enhanced


def _min_quad_side(image: np.ndarray) -> int:
  """Scale-dependent minimum grid side — avoids fixed 80px magic constant."""
  return max(60, int(min(image.shape[:2]) * 0.18))


# --- Stage 2: grid structure extraction ---
EXPECTED_GRID_LINES = 10
MIN_GRID_LINES = 9
MAX_SPACING_CV = 0.35
FAST_PATH_MIN_CONFIDENCE = 0.55
BORDER_EPSILON_RATIO = 0.02


@dataclass
class GridStructureMetrics:
  path: str
  vertical_line_count: int
  horizontal_line_count: int
  vertical_positions: list[float]
  horizontal_positions: list[float]
  spacing_variance: float
  clustering_confidence: float
  bounding_box: dict[str, float]

  def to_dict(self) -> dict:
    return {
      "path": self.path,
      "vertical_line_count": self.vertical_line_count,
      "horizontal_line_count": self.horizontal_line_count,
      "vertical_positions": [round(v, 1) for v in self.vertical_positions],
      "horizontal_positions": [round(v, 1) for v in self.horizontal_positions],
      "spacing_variance": round(self.spacing_variance, 4),
      "clustering_confidence": round(self.clustering_confidence, 3),
      "bounding_box": {k: round(v, 1) for k, v in self.bounding_box.items()},
    }


@dataclass
class GridDetectionResult:
  corners: np.ndarray
  path: str
  metrics: GridStructureMetrics

  def to_dict(self) -> dict:
    return {
      "corners": self.corners.tolist(),
      **self.metrics.to_dict(),
    }


def _detect_grid_corners(
  gray: np.ndarray,
  debug: DebugSession,
) -> tuple[GridDetectionResult | None, np.ndarray]:
  """FAST: reconstruct grid from line structure. ROBUST: contour fallback."""
  overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  median = float(np.median(blurred))
  edges = cv2.Canny(blurred, int(0.5 * median), int(1.5 * median))
  debug.save_image("04_edges.png", edges)

  fast = _extract_grid_from_lines(gray, edges)
  if fast is not None and fast.metrics.clustering_confidence >= FAST_PATH_MIN_CONFIDENCE:
    robust = _robust_contour_fallback(gray, overlay.copy())
    if robust is not None and _should_prefer_robust_over_fast(fast.corners, robust.corners):
      robust.metrics.path = "robust"
      _draw_quad(overlay, robust.corners, (0, 255, 0), thickness=3)
      _log_grid_detection(robust)
      _save_grid_detection_debug(debug, overlay, robust)
      return robust, overlay

    _draw_grid_lines_overlay(overlay, fast.metrics)
    _draw_quad(overlay, fast.corners, (0, 255, 0), thickness=3)
    _log_grid_detection(fast)
    _save_grid_detection_debug(debug, overlay, fast)
    return fast, overlay

  robust = _robust_contour_fallback(gray, overlay)
  if robust is None:
    _save_grid_detection_debug(debug, overlay, None)
    return None, overlay

  _draw_quad(overlay, robust.corners, (0, 255, 0), thickness=3)
  _log_grid_detection(robust)
  _save_grid_detection_debug(debug, overlay, robust)
  return robust, overlay


def _should_prefer_robust_over_fast(
  fast_corners: np.ndarray, robust_corners: np.ndarray
) -> bool:
  """Prefer perspective contour when line extraction yields an axis-aligned box."""
  if not _is_near_axis_aligned(fast_corners):
    return False
  if _is_near_axis_aligned(robust_corners):
    return False

  fast_area = float(
    cv2.contourArea(fast_corners.reshape(-1, 1, 2).astype(np.float32))
  )
  robust_area = float(
    cv2.contourArea(robust_corners.reshape(-1, 1, 2).astype(np.float32))
  )
  if fast_area <= 0 or robust_area <= 0:
    return False

  area_ratio = robust_area / fast_area
  return 0.75 <= area_ratio <= 1.35


def _is_near_axis_aligned(corners: np.ndarray, tolerance_deg: float = 10.0) -> bool:
  for index in range(4):
    point_a = corners[index]
    point_b = corners[(index + 1) % 4]
    angle = abs(np.degrees(np.arctan2(point_b[1] - point_a[1], point_b[0] - point_a[0])))
    angle = angle % 180
    deviation = min(angle, abs(angle - 90), abs(angle - 180))
    if deviation > tolerance_deg:
      return False
  return True


def _log_grid_detection(result: GridDetectionResult) -> None:
  m = result.metrics
  bbox = m.bounding_box
  logger.info(
    "Grid detection %s path: v_lines=%d h_lines=%d spacing_var=%.3f "
    "confidence=%.2f bbox=(%.0f,%.0f)-(%.0f,%.0f)",
    result.path.upper(),
    m.vertical_line_count,
    m.horizontal_line_count,
    m.spacing_variance,
    m.clustering_confidence,
    bbox["x0"],
    bbox["y0"],
    bbox["x1"],
    bbox["y1"],
  )


def _save_grid_detection_debug(
  debug: DebugSession,
  overlay: np.ndarray,
  result: GridDetectionResult | None,
) -> None:
  debug.save_image("04_candidate_contours.png", overlay)
  if result is not None:
    debug.save_json("04_detection.json", result.to_dict())
    debug.save_json("04_grid_structure.json", result.metrics.to_dict())


def _extract_grid_from_lines(
  gray: np.ndarray, edges: np.ndarray
) -> GridDetectionResult | None:
  """Detect grid via Hough lines, cluster to 10x10 boundaries, build corners."""
  height, width = gray.shape[:2]
  min_len = max(40, int(min(height, width) * 0.20))
  lines = cv2.HoughLinesP(
    edges,
    1,
    np.pi / 180,
    threshold=max(50, min_len // 4),
    minLineLength=min_len,
    maxLineGap=int(min(height, width) * 0.03),
  )
  if lines is None:
    return None

  h_segments: list[tuple[float, float, float, float]] = []
  v_segments: list[tuple[float, float, float, float]] = []
  for line in lines:
    x1, y1, x2, y2 = (float(v) for v in line[0])
    dx, dy = x2 - x1, y2 - y1
    length = float(np.hypot(dx, dy))
    if length < min_len * 0.45:
      continue
    angle = abs(np.degrees(np.arctan2(dy, dx)))
    if angle < 20 or angle > 160:
      h_segments.append((x1, y1, x2, y2))
    elif 70 < angle < 110:
      v_segments.append((x1, y1, x2, y2))

  raw_h = len(h_segments)
  raw_v = len(v_segments)
  if raw_h < MIN_GRID_LINES or raw_v < MIN_GRID_LINES:
    return None

  h_positions, h_clusters = _cluster_line_segments(
    h_segments, lambda seg: (seg[1] + seg[3]) / 2.0, EXPECTED_GRID_LINES, height
  )
  v_positions, v_clusters = _cluster_line_segments(
    v_segments, lambda seg: (seg[0] + seg[2]) / 2.0, EXPECTED_GRID_LINES, width
  )
  if h_positions is None or v_positions is None:
    return None

  valid, spacing_variance = _validate_grid_structure(h_positions, v_positions)
  if not valid:
    return None

  min_side = _min_quad_side(gray)
  grid_w = v_positions[-1] - v_positions[0]
  grid_h = h_positions[-1] - h_positions[0]
  if grid_w < min_side or grid_h < min_side:
    return None

  corners = _corners_from_grid_line_clusters(h_clusters, v_clusters)
  if corners is None:
    return None

  confidence = _clustering_confidence(raw_h, raw_v, spacing_variance)
  x_coords = [float(point[0]) for point in corners]
  y_coords = [float(point[1]) for point in corners]

  return GridDetectionResult(
    corners=corners,
    path="fast",
    metrics=GridStructureMetrics(
      path="fast",
      vertical_line_count=raw_v,
      horizontal_line_count=raw_h,
      vertical_positions=v_positions,
      horizontal_positions=h_positions,
      spacing_variance=spacing_variance,
      clustering_confidence=confidence,
      bounding_box={
        "x0": min(x_coords),
        "y0": min(y_coords),
        "x1": max(x_coords),
        "y1": max(y_coords),
      },
    ),
  )


def _cluster_line_segments(
  segments: list[tuple[float, float, float, float]],
  center_fn,
  expected: int,
  image_span: int,
) -> tuple[list[float] | None, list[list[tuple[float, float, float, float]]] | None]:
  centers = [center_fn(segment) for segment in segments]
  positions = _cluster_grid_lines(centers, expected, image_span)
  if positions is None:
    return None, None

  clusters: list[list[tuple[float, float, float, float]]] = [[] for _ in range(expected)]
  for segment in segments:
    center = center_fn(segment)
    nearest = min(range(expected), key=lambda index: abs(center - positions[index]))
    clusters[nearest].append(segment)

  return positions, clusters


def _corners_from_grid_line_clusters(
  h_clusters: list[list[tuple[float, float, float, float]]],
  v_clusters: list[list[tuple[float, float, float, float]]],
) -> np.ndarray | None:
  top_segments = _first_non_empty_cluster(h_clusters)
  bottom_segments = _last_non_empty_cluster(h_clusters)
  left_segments = _first_non_empty_cluster(v_clusters)
  right_segments = _last_non_empty_cluster(v_clusters)
  if None in (top_segments, bottom_segments, left_segments, right_segments):
    return None

  top_line = _fit_line_from_segments(top_segments)
  bottom_line = _fit_line_from_segments(bottom_segments)
  left_line = _fit_line_from_segments(left_segments)
  right_line = _fit_line_from_segments(right_segments)
  if (
    top_line is None
    or bottom_line is None
    or left_line is None
    or right_line is None
  ):
    return None

  top_left = _intersect_homogeneous_lines(top_line, left_line)
  top_right = _intersect_homogeneous_lines(top_line, right_line)
  bottom_right = _intersect_homogeneous_lines(bottom_line, right_line)
  bottom_left = _intersect_homogeneous_lines(bottom_line, left_line)
  if None in (top_left, top_right, bottom_right, bottom_left):
    return None

  return _order_corners(
    np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
  )


def _first_non_empty_cluster(
  clusters: list[list[tuple[float, float, float, float]]],
) -> list[tuple[float, float, float, float]] | None:
  for cluster in clusters:
    if cluster:
      return cluster
  return None


def _last_non_empty_cluster(
  clusters: list[list[tuple[float, float, float, float]]],
) -> list[tuple[float, float, float, float]] | None:
  for cluster in reversed(clusters):
    if cluster:
      return cluster
  return None


def _fit_line_from_segments(
  segments: list[tuple[float, float, float, float]],
) -> np.ndarray | None:
  if not segments:
    return None

  points: list[list[float]] = []
  for x1, y1, x2, y2 in segments:
    points.append([x1, y1])
    points.append([x2, y2])

  point_array = np.array(points, dtype=np.float32)
  line = cv2.fitLine(point_array, cv2.DIST_L2, 0, 0.01, 0.01)
  direction_x = float(line[0])
  direction_y = float(line[1])
  point_x = float(line[2])
  point_y = float(line[3])
  return np.cross(
    [point_x, point_y, 1.0],
    [point_x + direction_x, point_y + direction_y, 1.0],
  )


def _intersect_homogeneous_lines(
  line_a: np.ndarray, line_b: np.ndarray
) -> tuple[float, float] | None:
  intersection = np.cross(line_a, line_b)
  if abs(intersection[2]) < 1e-6:
    return None
  return float(intersection[0] / intersection[2]), float(intersection[1] / intersection[2])


def _cluster_grid_lines(
  positions: list[float], expected: int, image_span: int
) -> list[float] | None:
  """Merge nearby detections and reconstruct exactly `expected` grid boundaries."""
  if len(positions) < MIN_GRID_LINES:
    return None

  positions = sorted(positions)
  span = positions[-1] - positions[0]
  tolerance = max(8.0, span / (expected * 2.5) if span > 0 else image_span / expected / 3)
  merged = _merge_close_values(positions, tolerance)

  if len(merged) < MIN_GRID_LINES:
    return None

  if len(merged) == expected:
    return merged

  if len(merged) > expected:
    indices = np.linspace(0, len(merged) - 1, expected).astype(int)
    return [merged[i] for i in indices]

  first, last = merged[0], merged[-1]
  step = (last - first) / (expected - 1) if expected > 1 else 0.0
  return [first + i * step for i in range(expected)]


def _merge_close_values(values: list[float], tolerance: float) -> list[float]:
  clusters: list[list[float]] = [[values[0]]]
  for value in values[1:]:
    if value - clusters[-1][-1] <= tolerance:
      clusters[-1].append(value)
    else:
      clusters.append([value])
  return [float(np.median(cluster)) for cluster in clusters]


def _validate_grid_structure(
  h_lines: list[float], v_lines: list[float]
) -> tuple[bool, float]:
  if len(h_lines) != EXPECTED_GRID_LINES or len(v_lines) != EXPECTED_GRID_LINES:
    return False, 1.0

  h_spacings = np.diff(h_lines)
  v_spacings = np.diff(v_lines)
  if h_spacings.size == 0 or v_spacings.size == 0:
    return False, 1.0

  h_cv = float(np.std(h_spacings) / (np.mean(h_spacings) + 1e-6))
  v_cv = float(np.std(v_spacings) / (np.mean(v_spacings) + 1e-6))
  spacing_variance = (h_cv + v_cv) / 2.0

  if spacing_variance > MAX_SPACING_CV:
    return False, spacing_variance

  grid_w = v_lines[-1] - v_lines[0]
  grid_h = h_lines[-1] - h_lines[0]
  if grid_w <= 0 or grid_h <= 0:
    return False, spacing_variance

  aspect = min(grid_w, grid_h) / max(grid_w, grid_h)
  if aspect < 0.65:
    return False, spacing_variance

  return True, spacing_variance


def _clustering_confidence(
  raw_vertical: int, raw_horizontal: int, spacing_variance: float
) -> float:
  line_coverage = min(1.0, raw_vertical / EXPECTED_GRID_LINES) * min(
    1.0, raw_horizontal / EXPECTED_GRID_LINES
  )
  spacing_quality = max(0.0, 1.0 - spacing_variance / MAX_SPACING_CV)
  return line_coverage * spacing_quality


def _robust_contour_fallback(
  gray: np.ndarray, overlay: np.ndarray
) -> GridDetectionResult | None:
  """Fallback when line-structure extraction fails — largest valid quadrilateral."""
  detection_gray = _apply_clahe(gray)
  quads = _all_quadrilaterals(detection_gray)
  if not quads:
    return None

  height, width = gray.shape[:2]
  border_eps = max(8, int(min(width, height) * BORDER_EPSILON_RATIO))
  image_area = height * width

  for quad in quads:
    _draw_quad(overlay, quad, (255, 128, 0))

  def sort_key(quad: np.ndarray) -> tuple[float, float]:
    area = float(cv2.contourArea(quad.reshape(-1, 1, 2).astype(np.float32)))
    _, _, w, h = cv2.boundingRect(quad.astype(np.int32))
    squareness = min(w, h) / max(w, h) if max(w, h) else 0.0
    border_count = sum(
      1
      for x, y in quad
      if x <= border_eps
      or y <= border_eps
      or x >= width - border_eps
      or y >= height - border_eps
    )
    area_ratio = area / image_area if image_area else 1.0
    if border_count >= 3 and area_ratio > 0.92:
      return (-1.0, area * squareness)
    return (squareness, area)

  best = max(quads, key=sort_key)
  x_coords = [float(p[0]) for p in best]
  y_coords = [float(p[1]) for p in best]

  return GridDetectionResult(
    corners=best,
    path="robust",
    metrics=GridStructureMetrics(
      path="robust",
      vertical_line_count=0,
      horizontal_line_count=0,
      vertical_positions=[],
      horizontal_positions=[],
      spacing_variance=0.0,
      clustering_confidence=0.0,
      bounding_box={
        "x0": min(x_coords),
        "y0": min(y_coords),
        "x1": max(x_coords),
        "y1": max(y_coords),
      },
    ),
  )


def _draw_grid_lines_overlay(
  overlay: np.ndarray, metrics: GridStructureMetrics
) -> None:
  for y in metrics.horizontal_positions:
    cv2.line(overlay, (0, int(y)), (overlay.shape[1], int(y)), (0, 255, 255), 1)
  for x in metrics.vertical_positions:
    cv2.line(overlay, (int(x), 0), (int(x), overlay.shape[0]), (255, 0, 255), 1)


def _find_best_grid_corners(
  gray: np.ndarray,
  debug: DebugSession,
) -> tuple[np.ndarray | None, dict]:
  result, overlay = _detect_grid_corners(gray, debug)
  if result is None:
    return None, {"selected_overlay": overlay}
  return result.corners, {"selected_overlay": overlay, "detection": result.to_dict()}


def _draw_quad(
  image: np.ndarray,
  corners: np.ndarray,
  color: tuple[int, int, int],
  thickness: int = 2,
) -> None:
  pts = corners.reshape(-1, 1, 2).astype(np.int32)
  cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)


def _all_quadrilaterals(gray: np.ndarray) -> list[np.ndarray]:
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  quads: list[np.ndarray] = []
  min_side = _min_quad_side(gray)

  grad_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
  grad_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)
  gradient = cv2.convertScaleAbs(cv2.subtract(grad_x, grad_y))
  sources = [gradient]

  for use_adaptive in (False, True):
    if use_adaptive:
      block = _odd_block_size(blurred.shape)
      thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
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
    quads.extend(_quadrilaterals_from_binary(processed, min_side))

  median = float(np.median(blurred))
  low = int(max(0, 0.66 * median))
  high = int(min(255, 1.33 * median))
  edges = cv2.Canny(blurred, low, high)
  dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
  quads.extend(_quadrilaterals_from_binary(dilated, min_side))

  return quads


def _odd_block_size(shape: tuple[int, ...]) -> int:
  """Adaptive threshold block size relative to image — must be odd."""
  base = max(11, int(min(shape[:2]) / 40) | 1)
  return min(base, 31) if base % 2 else min(base + 1, 31)


def _quadrilaterals_from_binary(
  binary: np.ndarray, min_side: int
) -> list[np.ndarray]:
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
      if 0.65 < aspect < 1.5 and w >= min_side and h >= min_side:
        quads.append(points)
      break

  return quads


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


# --- Stage 3: grid validation gate ---
GRID_VALIDATION_MIN_LINES = 9
GRID_VALIDATION_MAX_LINES = 11
GRID_VALIDATION_MIN_CELLS = 72
GRID_VALIDATION_MAX_CELLS = 90
GRID_VALIDATION_MAX_EDGE_DENSITY = 0.25
GRID_VALIDATION_ASPECT_MIN = 0.75
GRID_VALIDATION_ASPECT_MAX = 1.25


def _prepare_warped_grid(
  gray: np.ndarray, corners: np.ndarray
) -> tuple[np.ndarray, list[int], list[int]]:
  warped = _warp_perspective(gray, corners)
  warped = cv2.fastNlMeansDenoising(warped, None, 10, 7, 21)
  h_lines, v_lines = _detect_grid_line_positions(warped)
  return warped, h_lines, v_lines


def _compute_warped_edge_density(warped: np.ndarray) -> float:
  blurred = cv2.GaussianBlur(warped, (3, 3), 0)
  median = float(np.median(blurred))
  edges = cv2.Canny(blurred, int(0.5 * median), int(1.5 * median))
  return float(np.count_nonzero(edges) / edges.size) if edges.size else 0.0


def _validate_detected_grid_structure(
  h_lines: list[int], v_lines: list[int], warped_image: np.ndarray
) -> tuple[bool, float]:
  """Return whether the warped grid is structurally valid and a confidence score."""
  details = _grid_validation_details(h_lines, v_lines, warped_image)
  return details["valid"], details["confidence"]


def _grid_validation_details(
  h_lines: list[int], v_lines: list[int], warped_image: np.ndarray
) -> dict:
  h_count = len(h_lines)
  v_count = len(v_lines)
  line_count_ok = (
    GRID_VALIDATION_MIN_LINES <= h_count <= GRID_VALIDATION_MAX_LINES
    and GRID_VALIDATION_MIN_LINES <= v_count <= GRID_VALIDATION_MAX_LINES
  )

  expected_cells = (h_count - 1) * (v_count - 1) if h_count and v_count else 0
  cells_ok = (
    GRID_VALIDATION_MIN_CELLS <= expected_cells <= GRID_VALIDATION_MAX_CELLS
  )

  h_spacings = np.diff(h_lines) if h_count > 1 else np.array([])
  v_spacings = np.diff(v_lines) if v_count > 1 else np.array([])
  if h_spacings.size and v_spacings.size:
    h_cv = float(np.std(h_spacings) / (np.mean(h_spacings) + 1e-6))
    v_cv = float(np.std(v_spacings) / (np.mean(v_spacings) + 1e-6))
    spacing_cv = (h_cv + v_cv) / 2.0
  else:
    spacing_cv = 1.0
  spacing_ok = spacing_cv <= MAX_SPACING_CV

  grid_w = float(v_lines[-1] - v_lines[0]) if v_count >= 2 else 0.0
  grid_h = float(h_lines[-1] - h_lines[0]) if h_count >= 2 else 0.0
  aspect_ratio = grid_w / grid_h if grid_h else 0.0
  aspect_ok = GRID_VALIDATION_ASPECT_MIN < aspect_ratio < GRID_VALIDATION_ASPECT_MAX

  edge_density = _compute_warped_edge_density(warped_image)
  edge_ok = edge_density <= GRID_VALIDATION_MAX_EDGE_DENSITY

  h_line_score = max(0.0, 1.0 - abs(h_count - EXPECTED_GRID_LINES) / 2.0)
  v_line_score = max(0.0, 1.0 - abs(v_count - EXPECTED_GRID_LINES) / 2.0)
  line_score = (h_line_score + v_line_score) / 2.0
  spacing_score = max(0.0, 1.0 - spacing_cv / MAX_SPACING_CV)
  aspect_score = max(0.0, 1.0 - abs(1.0 - aspect_ratio) / 0.25)
  edge_score = max(0.0, 1.0 - edge_density / GRID_VALIDATION_MAX_EDGE_DENSITY)
  confidence = (
    0.35 * line_score
    + 0.35 * spacing_score
    + 0.20 * aspect_score
    + 0.10 * edge_score
  )

  is_valid = line_count_ok and cells_ok and spacing_ok and aspect_ok and edge_ok
  return {
    "h_lines": h_count,
    "v_lines": v_count,
    "expected_cells": expected_cells,
    "spacing_cv": round(spacing_cv, 4),
    "aspect_ratio": round(aspect_ratio, 4),
    "edge_density": round(edge_density, 4),
    "valid": is_valid,
    "confidence": round(confidence, 3),
    "checks": {
      "line_count": line_count_ok,
      "cell_count": cells_ok,
      "spacing": spacing_ok,
      "aspect_ratio": aspect_ok,
      "edge_density": edge_ok,
    },
  }


def _log_grid_validation(details: dict) -> None:
  logger.info(
    "grid_validation: h_lines=%d v_lines=%d spacing_cv=%.2f "
    "aspect_ratio=%.2f edge_density=%.2f VALID=%s confidence=%.2f",
    details["h_lines"],
    details["v_lines"],
    details["spacing_cv"],
    details["aspect_ratio"],
    details["edge_density"],
    str(details["valid"]).lower(),
    details["confidence"],
  )


def _apply_grid_validation_gate(
  gray: np.ndarray,
  detection_enhanced: np.ndarray,
  corners: np.ndarray,
  detection_path: str,
  debug: DebugSession,
) -> tuple[np.ndarray, list[int], list[int], np.ndarray, dict]:
  """Warp, detect lines, validate; fall back to robust corners when fast fails."""
  warped, h_lines, v_lines = _prepare_warped_grid(gray, corners)
  is_valid, _ = _validate_detected_grid_structure(h_lines, v_lines, warped)
  validation = _grid_validation_details(h_lines, v_lines, warped)
  validation["detection_path"] = detection_path

  if not is_valid and detection_path == "fast":
    logger.info("Grid validation failed on FAST path — retrying with ROBUST fallback")
    overlay = cv2.cvtColor(detection_enhanced, cv2.COLOR_GRAY2BGR)
    robust = _robust_contour_fallback(detection_enhanced, overlay)
    if robust is not None:
      corners = robust.corners
      warped, h_lines, v_lines = _prepare_warped_grid(gray, corners)
      is_valid, _ = _validate_detected_grid_structure(h_lines, v_lines, warped)
      validation = _grid_validation_details(h_lines, v_lines, warped)
      validation["detection_path"] = "robust"
      validation["fallback_from"] = "fast"
      debug.save_image("05_robust_fallback_contour.png", overlay)

  _log_grid_validation(validation)
  debug.save_json("08_grid_validation.json", validation)

  if not is_valid:
    raise ValueError("No stable Sudoku grid detected")

  return warped, h_lines, v_lines, corners, validation


def _detect_grid_line_positions(warped: np.ndarray) -> tuple[list[int], list[int]]:
  """
  Find 10 horizontal and 10 vertical grid lines from projections.
  Reduces drift in lower rows/cols when the initial warp is slightly off.
  """
  binary = cv2.adaptiveThreshold(
    warped,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11,
    2,
  )
  h_positions = _find_line_positions(np.sum(binary, axis=1), GRID_SIZE, 10)
  v_positions = _find_line_positions(np.sum(binary, axis=0), GRID_SIZE, 10)
  return h_positions, v_positions


def _find_line_positions(
  projection: np.ndarray, image_size: int, expected: int
) -> list[int]:
  spacing = image_size / 9
  min_distance = max(8, int(spacing * 0.45))
  threshold = np.max(projection) * 0.30

  peaks: list[int] = []
  last = -min_distance
  for i, value in enumerate(projection):
    if value > threshold and i - last >= min_distance:
      peaks.append(i)
      last = i

  uniform = [int(round(i * image_size / 9)) for i in range(expected)]

  if len(peaks) < 4:
    return uniform

  # Snap detected peaks to a uniform 10-line model via linear fit.
  if len(peaks) >= expected - 2:
    peaks = _select_best_spaced_peaks(peaks, expected, min_distance)

  if len(peaks) < 4:
    return uniform

  # Interpolate missing lines between first and last peak.
  first, last = peaks[0], peaks[-1]
  step = (last - first) / (expected - 1) if expected > 1 else spacing
  return [int(round(first + i * step)) for i in range(expected)]


def _select_best_spaced_peaks(
  peaks: list[int], expected: int, min_distance: int
) -> list[int]:
  """Keep peaks that best match expected grid spacing."""
  if len(peaks) <= expected:
    return peaks

  target_spacing = (peaks[-1] - peaks[0]) / max(expected - 1, 1)
  selected = [peaks[0]]
  for peak in peaks[1:]:
    if peak - selected[-1] >= min_distance * 0.7:
      selected.append(peak)
  if len(selected) > expected:
    # Subsample to expected count.
    indices = np.linspace(0, len(selected) - 1, expected).astype(int)
    selected = [selected[i] for i in indices]
  return selected


def _extract_cell(
  warped: np.ndarray,
  row: int,
  col: int,
  h_lines: list[int],
  v_lines: list[int],
) -> tuple[np.ndarray, dict[str, int | float]]:
  y1, y2 = h_lines[row], h_lines[row + 1]
  x1, x2 = v_lines[col], v_lines[col + 1]
  cell_height = y2 - y1
  cell_width = x2 - x1
  margin_y = max(1, int(cell_height * CELL_MARGIN_RATIO))
  margin_x = max(1, int(cell_width * CELL_MARGIN_RATIO))
  crop_y1 = y1 + margin_y
  crop_y2 = y2 - margin_y
  crop_x1 = x1 + margin_x
  crop_x2 = x2 - margin_x
  cell = warped[crop_y1:crop_y2, crop_x1:crop_x2]
  meta: dict[str, int | float] = {
    "x1": crop_x1,
    "y1": crop_y1,
    "x2": crop_x2,
    "y2": crop_y2,
    "margin_x": margin_x,
    "margin_y": margin_y,
    "margin_ratio": CELL_MARGIN_RATIO,
  }
  return cell, meta


def _ocr_cells(
  warped: np.ndarray,
  h_lines: list[int],
  v_lines: list[int],
  debug: DebugSession,
) -> list[list[int]]:
  board: list[list[int]] = []
  preprocess_reports: list[dict] = []

  for row in range(9):
    row_values: list[int] = []
    for col in range(9):
      cell, coords = _extract_cell(warped, row, col, h_lines, v_lines)
      debug.save_image(f"cells/raw/r{row}_c{col}.png", cell)

      result = _recognize_cell(cell, row, col, coords, debug)
      row_values.append(result.digit)

      debug.save_image(f"cells/processed/r{row}_c{col}.png", result.processed)
      report = {
        "digit": result.digit,
        "confidence": result.confidence,
        "votes": result.votes,
        "empty": result.empty,
        "ambiguous": result.ambiguous,
      }
      if result.metrics is not None:
        report = {**result.metrics.to_dict(), **report}
      preprocess_reports.append(report)
      debug.log_cell(report)

    board.append(row_values)

  debug.save_json("09_cell_preprocessing.json", preprocess_reports)
  return board


def _recognize_cell(
  cell: np.ndarray,
  row: int,
  col: int,
  coords: dict[str, int | float],
  debug: DebugSession,
) -> CellOcrResult:
  blank = _blank_cell_image()
  if cell.size == 0:
    metrics = _empty_metrics(row, col, coords)
    return CellOcrResult(0, 0, blank, {}, True, False, metrics)

  preprocessed_variants = _preprocess_cell_variants(cell, row, col, coords, debug)
  non_empty = [p for p in preprocessed_variants if not p.empty]
  if not non_empty:
    return CellOcrResult(
      0,
      0,
      preprocessed_variants[0].final,
      {},
      True,
      False,
      preprocessed_variants[0].metrics,
    )

  primary = max(non_empty, key=lambda p: p.metrics.largest_component_area)

  if primary.suspicious_border and BORDER_DIGIT_ACTION == "discard":
    metrics = primary.metrics
    metrics.empty = True
    return CellOcrResult(0, 0, primary.final, {}, True, False, metrics)

  votes: dict[int, int] = {}
  conf_by_digit: dict[int, int] = {}

  seen_images: set[bytes] = set()
  for preprocessed in non_empty:
    if preprocessed.suspicious_border and BORDER_DIGIT_ACTION == "discard":
      continue
    for final_image in _ocr_input_variants(preprocessed.final):
      image_key = final_image.tobytes()
      if image_key in seen_images:
        continue
      seen_images.add(image_key)
      pil_image = Image.fromarray(final_image)
      for config in TESSERACT_CONFIGS:
        digit, conf = _tesseract_read_with_confidence(pil_image, config)
        if digit == 0:
          continue
        if (
          preprocessed.suspicious_border
          and BORDER_DIGIT_ACTION == "lower_confidence"
        ):
          conf = max(0, conf - BORDER_DIGIT_CONFIDENCE_PENALTY)
          if conf < MIN_ACCEPT_CONFIDENCE:
            continue
        votes[digit] = votes.get(digit, 0) + max(conf, 1)
        conf_by_digit[digit] = max(conf_by_digit.get(digit, 0), conf)

  if not votes:
    return CellOcrResult(
      0, 0, primary.final, {}, True, False, primary.metrics
    )

  ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
  top_digit, top_vote = ranked[0]
  runner_up_vote = ranked[1][1] if len(ranked) > 1 else 0
  top_conf = conf_by_digit.get(top_digit, 0)

  ambiguous = False
  if runner_up_vote > 0:
    margin = top_vote / runner_up_vote
    if margin < MIN_VOTE_MARGIN_RATIO:
      ambiguous = True
    elif top_conf < MIN_ACCEPT_CONFIDENCE and margin < 1.8:
      ambiguous = True

  if ambiguous:
    return CellOcrResult(
      0, top_conf, primary.final, votes, False, True, primary.metrics
    )

  return CellOcrResult(
    top_digit, top_conf, primary.final, votes, False, False, primary.metrics
  )


def _empty_metrics(
  row: int, col: int, coords: dict[str, int | float]
) -> CellPreprocessMetrics:
  return CellPreprocessMetrics(
    row=row,
    col=col,
    coords={k: int(v) for k, v in coords.items() if k in ("x1", "y1", "x2", "y2")},
    margin_ratio=float(coords.get("margin_ratio", CELL_MARGIN_RATIO)),
    margin_x=int(coords.get("margin_x", 0)),
    margin_y=int(coords.get("margin_y", 0)),
    foreground_pixels=0,
    foreground_percent=0.0,
    component_count=0,
    largest_component_area=0,
    bounding_box=None,
    suspicious_border=False,
    empty=True,
  )


def _preprocess_cell_variants(
  cell: np.ndarray,
  row: int,
  col: int,
  coords: dict[str, int | float],
  debug: DebugSession,
) -> list[CellPreprocessResult]:
  original = cv2.resize(
    cell, (CELL_WORKING_SIZE, CELL_WORKING_SIZE), interpolation=cv2.INTER_CUBIC
  )
  clahe = _apply_clahe(original)
  normalized = _normalize_illumination(clahe)
  blurred = cv2.GaussianBlur(normalized, (3, 3), 0)

  _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  block = _odd_block_size(blurred.shape)
  adaptive = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    block,
    2,
  )

  results = [
    _finalize_preprocessed_cell(original, _normalize_binary(otsu), row, col, coords),
    _finalize_preprocessed_cell(
      original, _normalize_binary(adaptive), row, col, coords
    ),
  ]

  if debug.enabled and (row, col) in DEBUG_SAMPLE_CELLS:
    primary = max(
      results,
      key=lambda item: item.metrics.largest_component_area if not item.empty else -1,
    )
    _save_cell_debug_stages(debug, row, col, primary)

  return results


def _finalize_preprocessed_cell(
  original: np.ndarray,
  threshold: np.ndarray,
  row: int,
  col: int,
  coords: dict[str, int | float],
) -> CellPreprocessResult:
  opened = _apply_opening(threshold)
  border_removed, component_count = _remove_border_components(opened)
  metrics = _measure_cell_foreground(
    border_removed, row, col, coords, component_count
  )
  empty = _is_empty_from_metrics(metrics)
  metrics.empty = empty

  if empty:
    blank = _blank_cell_image()
    return CellPreprocessResult(
      original=original,
      threshold=threshold,
      border_removed=border_removed,
      centered=blank,
      final=blank,
      metrics=metrics,
      empty=True,
      suspicious_border=False,
    )

  cropped, _ = _crop_foreground_tight(border_removed)
  inner_bbox = _foreground_bbox(cropped)
  suspicious = False
  if inner_bbox is not None:
    suspicious = _is_suspicious_border_digit(
      inner_bbox, cropped.shape[1], cropped.shape[0], BORDER_TOUCH_THRESHOLD
    )
  metrics.bounding_box = inner_bbox
  metrics.suspicious_border = suspicious

  centered = _center_digit_in_square(cropped, CENTERED_SQUARE_SIZE)
  final = _upscale_for_ocr(centered, OCR_CANVAS_SIZE)

  return CellPreprocessResult(
    original=original,
    threshold=threshold,
    border_removed=border_removed,
    centered=centered,
    final=final,
    metrics=metrics,
    empty=False,
    suspicious_border=suspicious,
  )


def _apply_opening(binary: np.ndarray) -> np.ndarray:
  if OPEN_KERNEL_SIZE <= 0:
    return binary
  kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT, (OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE)
  )
  return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)


def _remove_border_components(
  binary: np.ndarray,
) -> tuple[np.ndarray, int]:
  """Drop foreground components that touch any image edge (grid lines)."""
  foreground = (binary == 0).astype(np.uint8)
  num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    foreground, connectivity=8
  )
  cleaned = np.full_like(binary, 255)
  kept = 0

  for label in range(1, num_labels):
    area = int(stats[label, cv2.CC_STAT_AREA])
    if area < MIN_COMPONENT_AREA:
      continue

    component_mask = labels == label
    touches_border = (
      np.any(component_mask[0, :])
      or np.any(component_mask[-1, :])
      or np.any(component_mask[:, 0])
      or np.any(component_mask[:, -1])
    )
    if touches_border:
      continue

    cleaned[component_mask] = 0
    kept += 1

  return cleaned, kept


def _measure_cell_foreground(
  binary: np.ndarray,
  row: int,
  col: int,
  coords: dict[str, int | float],
  component_count: int,
) -> CellPreprocessMetrics:
  foreground_pixels = int(np.count_nonzero(binary == 0))
  total_pixels = binary.size
  foreground_percent = foreground_pixels / total_pixels if total_pixels else 0.0

  largest_area = 0
  largest_bbox: dict[str, int] | None = None
  inverted = 255 - binary
  contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if contours:
    largest = max(contours, key=cv2.contourArea)
    largest_area = int(cv2.contourArea(largest))
    x, y, w, h = cv2.boundingRect(largest)
    largest_bbox = {"x": x, "y": y, "w": w, "h": h}

  return CellPreprocessMetrics(
    row=row,
    col=col,
    coords={k: int(v) for k, v in coords.items() if k in ("x1", "y1", "x2", "y2")},
    margin_ratio=float(coords.get("margin_ratio", CELL_MARGIN_RATIO)),
    margin_x=int(coords.get("margin_x", 0)),
    margin_y=int(coords.get("margin_y", 0)),
    foreground_pixels=foreground_pixels,
    foreground_percent=foreground_percent,
    component_count=component_count,
    largest_component_area=largest_area,
    bounding_box=largest_bbox,
    suspicious_border=False,
    empty=False,
  )


def _is_empty_from_metrics(metrics: CellPreprocessMetrics) -> bool:
  if metrics.foreground_percent < EMPTY_FOREGROUND_RATIO:
    return True
  if metrics.largest_component_area < MIN_LARGEST_COMPONENT_AREA:
    return True
  if metrics.component_count == 0:
    return True
  bbox = metrics.bounding_box
  if bbox is None:
    return True
  return bbox["h"] < 6 or bbox["w"] < 3


def _foreground_bbox(binary: np.ndarray) -> dict[str, int] | None:
  inverted = 255 - binary
  contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if not contours:
    return None
  x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
  return {"x": x, "y": y, "w": w, "h": h}


def _crop_foreground_tight(
  binary: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
  inverted = 255 - binary
  contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if not contours:
    return binary, {"x": 0, "y": 0, "w": binary.shape[1], "h": binary.shape[0]}

  x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
  pad = max(2, int(min(w, h) * 0.08))
  x1 = max(0, x - pad)
  y1 = max(0, y - pad)
  x2 = min(binary.shape[1], x + w + pad)
  y2 = min(binary.shape[0], y + h + pad)
  cropped = binary[y1:y2, x1:x2]
  return cropped, {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}


def _is_suspicious_border_digit(
  bbox: dict[str, int], crop_w: int, crop_h: int, threshold: int
) -> bool:
  """Flag line-like foreground hugging the crop edge — typical grid-line residue."""
  x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
  touches_top = y <= threshold
  touches_bottom = y + h >= crop_h - threshold
  touches_left = x <= threshold
  touches_right = x + w >= crop_w - threshold

  if (touches_top or touches_bottom) and h <= 8 and w >= crop_w * 0.35:
    return True
  if (touches_left or touches_right) and w <= 8 and h >= crop_h * 0.35:
    return True
  return False


def _center_digit_in_square(binary: np.ndarray, square_size: int) -> np.ndarray:
  h, w = binary.shape
  scale = square_size / max(h, w, 1)
  new_w = max(1, int(w * scale))
  new_h = max(1, int(h * scale))
  resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)

  canvas = np.full((square_size, square_size), 255, dtype=np.uint8)
  y_off = (square_size - new_h) // 2
  x_off = (square_size - new_w) // 2
  canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
  return canvas


def _upscale_for_ocr(binary: np.ndarray, canvas_size: int) -> np.ndarray:
  if binary.shape[0] >= canvas_size:
    return binary
  return _center_digit_in_square(binary, canvas_size)


def _ocr_input_variants(binary: np.ndarray) -> list[np.ndarray]:
  if binary.shape[0] < OCR_CANVAS_SIZE:
    return [binary, _center_digit_in_square(binary, OCR_CANVAS_SIZE)]
  return [binary]


def _save_cell_debug_stages(
  debug: DebugSession, row: int, col: int, result: CellPreprocessResult
) -> None:
  prefix = f"cells/detail/r{row}_c{col}"
  debug.save_image(f"{prefix}/01_original.png", result.original)
  debug.save_image(f"{prefix}/02_threshold.png", result.threshold)
  debug.save_image(f"{prefix}/03_border_removed.png", result.border_removed)
  debug.save_image(f"{prefix}/04_centered_digit.png", result.centered)
  debug.save_image(f"{prefix}/05_final_input.png", result.final)


def _normalize_illumination(gray: np.ndarray) -> np.ndarray:
  """Reduce shadows by dividing by a heavily blurred background estimate."""
  background = cv2.GaussianBlur(gray, (0, 0), sigmaX=gray.shape[0] / 4)
  normalized = cv2.divide(gray, background, scale=128)
  return np.clip(normalized, 0, 255).astype(np.uint8)


def _normalize_binary(binary: np.ndarray) -> np.ndarray:
  if np.mean(binary) < 127:
    binary = 255 - binary
  return binary


def _blank_cell_image() -> np.ndarray:
  return np.full((OCR_CANVAS_SIZE, OCR_CANVAS_SIZE), 255, dtype=np.uint8)


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


def _draw_grid_lines_on(image: np.ndarray) -> np.ndarray:
  if len(image.shape) == 2:
    preview = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
  else:
    preview = image.copy()
  for i in range(10):
    thickness = 2 if i % 3 == 0 else 1
    pos = i * CELL_SIZE
    cv2.line(preview, (pos, 0), (pos, GRID_SIZE), (0, 0, 255), thickness)
    cv2.line(preview, (0, pos), (GRID_SIZE, pos), (0, 0, 255), thickness)
  return preview
