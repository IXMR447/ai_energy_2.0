from __future__ import annotations

import argparse
import math
import re

# Backport argparse.BooleanOptionalAction for Python < 3.9.
# This is the CPython 3.9+ implementation, added so the pipeline also runs on
# Python 3.8 (the project's bundled venvs are 3.8).
if not hasattr(argparse, "BooleanOptionalAction"):
    class _BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None, type=None,
                     choices=None, required=False, help=None, metavar=None):
            _option_strings = []
            for option_string in option_strings:
                _option_strings.append(option_string)
                if option_string.startswith("--"):
                    option_string = "--no-" + option_string[2:]
                    _option_strings.append(option_string)
            if help is not None and default is not None and default is not argparse.SUPPRESS:
                help += " (default: %(default)s)"
            super().__init__(
                option_strings=_option_strings,
                dest=dest,
                nargs=0,
                default=default,
                type=type,
                choices=choices,
                required=required,
                help=help,
                metavar=metavar,
            )

        def __call__(self, parser, namespace, values, option_string=None):
            if option_string in self.option_strings:
                setattr(namespace, self.dest, not option_string.startswith("--no-"))

        def format_usage(self):
            return " | ".join(self.option_strings)

    argparse.BooleanOptionalAction = _BooleanOptionalAction
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_CONFIG_PATH = Path("config.yaml")


@dataclass
class Detection:
    frame: str
    frame_index: int
    class_id: int
    class_name: str
    confidence: float
    contour: np.ndarray
    xc_px: float
    yc_px: float
    area_px2: float
    perimeter_px: float
    circularity: float
    diameter_px: float
    track_id: int | None = None
    mean_color: np.ndarray | None = None


def natural_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def list_images(images_dir: Path) -> list[Path]:
    images = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(images, key=natural_key)


def read_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def normalize_names(names: dict[int, str] | list[str]) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(idx): str(name) for idx, name in names.items()}
    return {idx: str(name) for idx, name in enumerate(names)}


def parse_target_classes(value: str | None, names: dict[int, str] | list[str]) -> set[int] | None:
    if not value:
        return None

    wanted = {item.strip() for item in value.split(",") if item.strip()}
    if not wanted:
        return None

    normalized_names = normalize_names(names)
    selected: set[int] = set()
    name_to_id = {name: idx for idx, name in normalized_names.items()}
    for item in wanted:
        if item.isdigit():
            selected.add(int(item))
        elif item in name_to_id:
            selected.add(name_to_id[item])
        else:
            known = ", ".join(f"{idx}:{name}" for idx, name in normalized_names.items())
            raise ValueError(f"Unknown target class '{item}'. Known classes: {known}")
    return selected


def polygon_to_contour(poly: np.ndarray) -> np.ndarray | None:
    if poly is None or len(poly) < 3:
        return None
    contour = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    return contour


def measure_contour(contour: np.ndarray) -> tuple[float, float, float, float, float, float]:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if area <= 0 or perimeter <= 0:
        return math.nan, math.nan, area, perimeter, math.nan, math.nan

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        points = contour.reshape(-1, 2)
        xc = float(points[:, 0].mean())
        yc = float(points[:, 1].mean())
    else:
        xc = float(moments["m10"] / moments["m00"])
        yc = float(moments["m01"] / moments["m00"])

    circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
    diameter = float(math.sqrt(4.0 * area / math.pi))
    return xc, yc, area, perimeter, circularity, diameter


def mask_mean_color(image: np.ndarray | None, contour: np.ndarray | None) -> np.ndarray | None:
    """Mean BGR color of the pixels inside a mask contour, or None if unavailable."""
    if image is None or contour is None or len(contour) < 3:
        return None
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [np.asarray(contour, dtype=np.int32)], -1, 255, -1)
    pixels = image[mask > 0]
    if pixels.size == 0:
        return None
    return pixels.astype(np.float32).mean(axis=0)


def mask_iou(contour_a: np.ndarray | None, contour_b: np.ndarray | None) -> float:
    """Intersection-over-union of two mask contours, in the shared image space."""
    if contour_a is None or contour_b is None:
        return 0.0
    ca = np.asarray(contour_a, dtype=np.int32)
    cb = np.asarray(contour_b, dtype=np.int32)
    if len(ca) < 3 or len(cb) < 3:
        return 0.0
    xa, ya, wa, ha = cv2.boundingRect(ca)
    xb, yb, wb, hb = cv2.boundingRect(cb)
    x = min(xa, xb)
    y = min(ya, yb)
    w = max(xa + wa, xb + wb) - x
    h = max(ya + ha, yb + hb) - y
    if w <= 0 or h <= 0:
        return 0.0
    shift = np.array([x, y], dtype=np.int32).reshape(1, 1, 2)
    mask_a = np.zeros((h, w), dtype=np.uint8)
    mask_b = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask_a, [ca - shift], -1, 255, -1)
    cv2.drawContours(mask_b, [cb - shift], -1, 255, -1)
    inter = int(np.count_nonzero(mask_a & mask_b))
    union = int(np.count_nonzero(mask_a | mask_b))
    return inter / union if union else 0.0


def _color_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """BGR L2 distance between two mean mask colors; 0 when either is missing."""
    if a is None or b is None:
        return 0.0
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 0.0
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def detection_from_polygon(
    frame: str,
    frame_index: int,
    names: dict[int, str],
    class_id: int,
    confidence: float,
    poly: np.ndarray,
    track_id: int | None = None,
    image: np.ndarray | None = None,
) -> Detection | None:
    contour = polygon_to_contour(poly)
    if contour is None:
        return None

    xc, yc, area, perimeter, circularity, diameter = measure_contour(contour)
    if not np.isfinite(area) or area <= 0:
        return None

    return Detection(
        frame=frame,
        frame_index=frame_index,
        class_id=int(class_id),
        class_name=str(names.get(int(class_id), class_id)),
        confidence=float(confidence),
        contour=contour,
        xc_px=xc,
        yc_px=yc,
        area_px2=area,
        perimeter_px=perimeter,
        circularity=circularity,
        diameter_px=diameter,
        track_id=track_id,
        mean_color=mask_mean_color(image, contour),
    )


def run_inference(
    model: YOLO,
    image_paths: list[Path],
    target_class_ids: set[int] | None,
    conf: float,
    imgsz: int,
    device: str | None,
) -> list[Detection]:
    detections: list[Detection] = []
    names = normalize_names(model.names)

    for frame_index, image_path in enumerate(image_paths):
        image = read_image(image_path)
        results = model.predict(
            source=str(image_path),
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        result = results[0]
        if result.masks is None or result.boxes is None:
            continue

        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        polygons = result.masks.xy

        for class_id, confidence, poly in zip(class_ids, confidences, polygons):
            if target_class_ids is not None and int(class_id) not in target_class_ids:
                continue

            detection = detection_from_polygon(
                frame=image_path.name,
                frame_index=frame_index,
                names=names,
                class_id=int(class_id),
                confidence=float(confidence),
                poly=poly,
                image=image,
            )
            if detection is not None:
                detections.append(detection)
    return detections


def run_bytetrack(
    model: YOLO,
    image_paths: list[Path],
    target_class_ids: set[int] | None,
    conf: float,
    imgsz: int,
    device: str | None,
    tracker_config: str,
) -> list[Detection]:
    detections: list[Detection] = []
    names = normalize_names(model.names)

    for frame_index, image_path in enumerate(image_paths):
        results = model.track(
            source=str(image_path),
            conf=conf,
            imgsz=imgsz,
            device=device,
            tracker=tracker_config,
            persist=True,
            verbose=False,
        )
        result = results[0]
        if result.masks is None or result.boxes is None:
            continue

        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        polygons = result.masks.xy
        if result.boxes.id is None:
            track_ids: list[int | None] = [None] * len(class_ids)
        else:
            track_ids = [int(value) for value in result.boxes.id.cpu().numpy().astype(int)]

        for class_id, confidence, poly, track_id in zip(class_ids, confidences, polygons, track_ids):
            if target_class_ids is not None and int(class_id) not in target_class_ids:
                continue

            detection = detection_from_polygon(
                frame=image_path.name,
                frame_index=frame_index,
                names=names,
                class_id=int(class_id),
                confidence=float(confidence),
                poly=poly,
                track_id=track_id,
            )
            if detection is not None:
                detections.append(detection)
    return detections


def assign_tracks(detections: list[Detection], max_distance_px: float) -> None:
    existing_ids = [det.track_id for det in detections if det.track_id is not None]
    next_track_id = max(existing_ids, default=0) + 1
    active: dict[int, Detection] = {}

    frame_indices = sorted({det.frame_index for det in detections})
    by_frame = {
        frame_index: [det for det in detections if det.frame_index == frame_index]
        for frame_index in frame_indices
    }

    previous_frame: int | None = None
    for frame_index in frame_indices:
        current = by_frame[frame_index]
        if previous_frame is None or frame_index != previous_frame + 1:
            active = {}

        unmatched_tracks = set(active.keys())
        unmatched_dets = {idx for idx, det in enumerate(current) if det.track_id is None}
        candidates: list[tuple[float, int, int]] = []

        for track_id, prev in active.items():
            for det_idx, det in enumerate(current):
                if prev.class_id != det.class_id:
                    continue
                dist = math.hypot(det.xc_px - prev.xc_px, det.yc_px - prev.yc_px)
                if dist <= max_distance_px:
                    candidates.append((dist, track_id, det_idx))

        for _, track_id, det_idx in sorted(candidates, key=lambda item: item[0]):
            if track_id not in unmatched_tracks or det_idx not in unmatched_dets:
                continue
            current[det_idx].track_id = track_id
            unmatched_tracks.remove(track_id)
            unmatched_dets.remove(det_idx)

        for det_idx in sorted(unmatched_dets):
            current[det_idx].track_id = next_track_id
            next_track_id += 1

        active = {det.track_id: det for det in current if det.track_id is not None}
        previous_frame = frame_index


def suppress_overlapping_detections(
    detections: list[Detection],
    iou_threshold: float,
    center_ratio: float,
) -> list[Detection]:
    """Remove near-duplicate / halo detections within each frame.

    YOLO sometimes emits several overlapping masks for one droplet (duplicates)
    or a dimmer glow/halo ring around a bright droplet. Within each frame we keep
    only the highest-confidence detection of each overlapping group, which avoids
    both splitting one physical droplet into several track ids and tracking a
    halo as a separate droplet.
    """
    by_frame: dict[int, list[Detection]] = {}
    for det in detections:
        by_frame.setdefault(det.frame_index, []).append(det)

    kept: list[Detection] = []
    for frame_index in sorted(by_frame):
        frame_dets = sorted(by_frame[frame_index], key=lambda d: d.confidence, reverse=True)
        survivors: list[Detection] = []
        for det in frame_dets:
            suppressed = False
            for other in survivors:
                if det.class_id != other.class_id:
                    continue
                dist = math.hypot(det.xc_px - other.xc_px, det.yc_px - other.yc_px)
                if iou_threshold > 0 and mask_iou(det.contour, other.contour) > iou_threshold:
                    suppressed = True
                    break
                if center_ratio > 0 and dist < center_ratio * max(det.diameter_px, other.diameter_px):
                    suppressed = True
                    break
            if not suppressed:
                survivors.append(det)
        kept.extend(survivors)

    return kept


def assign_tracks_motion(
    detections: list[Detection],
    max_distance_px: float,
    max_gap: int,
    max_size_ratio: float = 3.0,
    max_color_distance: float = 70.0,
) -> None:
    if not detections:
        return

    next_track_id = 1
    active: dict[int, list[Detection]] = {}
    by_frame: dict[int, list[Detection]] = {}
    for det in detections:
        by_frame.setdefault(det.frame_index, []).append(det)

    for frame_index in range(min(by_frame), max(by_frame) + 1):
        current = by_frame.get(frame_index, [])
        active = {
            track_id: track
            for track_id, track in active.items()
            if frame_index - track[-1].frame_index <= max_gap + 1
        }

        unmatched_tracks = set(active.keys())
        unmatched_dets = set(range(len(current)))
        candidates: list[tuple[float, int, int]] = []

        for track_id, track in active.items():
            last = track[-1]
            for det_idx, det in enumerate(current):
                if last.class_id != det.class_id:
                    continue
                frame_gap = max(1, det.frame_index - last.frame_index)
                pred_x, pred_y = _predict_next_position(track, frame_gap)
                distance = math.hypot(det.xc_px - pred_x, det.yc_px - pred_y)
                allowed_distance = max_distance_px * math.sqrt(frame_gap)
                if distance > allowed_distance:
                    continue

                size_ratio = 1.0
                if last.area_px2 > 0 and det.area_px2 > 0:
                    size_ratio = max(last.area_px2, det.area_px2) / min(last.area_px2, det.area_px2)
                if size_ratio > max_size_ratio:
                    continue

                color_dist = _color_distance(last.mean_color, det.mean_color)
                if max_color_distance > 0 and color_dist > max_color_distance:
                    continue

                s_pos = distance / allowed_distance if allowed_distance > 0 else 0.0
                s_size = abs(math.log(size_ratio)) / math.log(max_size_ratio) if max_size_ratio > 1 else 0.0
                s_color = color_dist / max_color_distance if max_color_distance > 0 else 0.0
                s_iou = 0.0
                if frame_gap == 1:
                    s_iou = 1.0 - mask_iou(last.contour, det.contour)

                score = s_pos + s_size + s_color + s_iou
                candidates.append((score, track_id, det_idx))

        for _, track_id, det_idx in sorted(candidates, key=lambda item: item[0]):
            if track_id not in unmatched_tracks or det_idx not in unmatched_dets:
                continue
            det = current[det_idx]
            det.track_id = track_id
            active[track_id].append(det)
            unmatched_tracks.remove(track_id)
            unmatched_dets.remove(det_idx)

        for det_idx in sorted(unmatched_dets):
            det = current[det_idx]
            det.track_id = next_track_id
            active[next_track_id] = [det]
            next_track_id += 1


def _track_groups(detections: list[Detection]) -> dict[int, list[Detection]]:
    groups: dict[int, list[Detection]] = {}
    for det in detections:
        if det.track_id is None:
            continue
        groups.setdefault(int(det.track_id), []).append(det)
    for group in groups.values():
        group.sort(key=lambda det: det.frame_index)
    return groups


def _predict_next_position(track: list[Detection], frame_gap: int) -> tuple[float, float]:
    last = track[-1]
    if len(track) < 2:
        return last.xc_px, last.yc_px

    prev = track[-2]
    dt = max(1, last.frame_index - prev.frame_index)
    vx = (last.xc_px - prev.xc_px) / dt
    vy = (last.yc_px - prev.yc_px) / dt
    return last.xc_px + vx * frame_gap, last.yc_px + vy * frame_gap


def link_tracks_across_gaps(
    detections: list[Detection],
    max_gap: int,
    max_link_distance_px: float,
    max_size_ratio: float = 3.0,
    max_color_distance: float = 70.0,
) -> int:
    if max_gap <= 0:
        return 0

    links_made = 0
    while True:
        groups = _track_groups(detections)
        candidates: list[tuple[float, int, int]] = []
        track_ids = sorted(groups)

        for source_id in track_ids:
            source = groups[source_id]
            source_end = source[-1]
            for target_id in track_ids:
                if source_id == target_id:
                    continue
                target = groups[target_id]
                target_start = target[0]
                frame_gap = target_start.frame_index - source_end.frame_index
                missing_frames = frame_gap - 1
                if missing_frames < 1 or missing_frames > max_gap:
                    continue
                if source_end.class_id != target_start.class_id:
                    continue

                pred_x, pred_y = _predict_next_position(source, frame_gap)
                distance = math.hypot(target_start.xc_px - pred_x, target_start.yc_px - pred_y)
                allowed_distance = max_link_distance_px * math.sqrt(frame_gap)
                if distance > allowed_distance:
                    continue

                size_ratio = 1.0
                if source_end.area_px2 > 0 and target_start.area_px2 > 0:
                    size_ratio = max(source_end.area_px2, target_start.area_px2) / min(source_end.area_px2, target_start.area_px2)
                if size_ratio > max_size_ratio:
                    continue

                color_dist = _color_distance(source_end.mean_color, target_start.mean_color)
                if max_color_distance > 0 and color_dist > max_color_distance:
                    continue

                s_pos = distance / allowed_distance if allowed_distance > 0 else 0.0
                s_size = abs(math.log(size_ratio)) / math.log(max_size_ratio) if max_size_ratio > 1 else 0.0
                s_color = color_dist / max_color_distance if max_color_distance > 0 else 0.0
                score = s_pos + s_size + s_color
                candidates.append((score, source_id, target_id))

        if not candidates:
            break

        _, source_id, target_id = min(candidates, key=lambda item: item[0])
        for det in detections:
            if det.track_id == target_id:
                det.track_id = source_id
        links_made += 1

    return links_made


def detections_to_dataframe(
    detections: list[Detection],
    fps: float,
    px_size_mm: float | None,
    interpolate_missing: bool,
    fit_tracks: bool,
    track_fit_degree: int,
    use_fitted_for_speed: bool,
) -> pd.DataFrame:
    rows = []
    for det in detections:
        scale = px_size_mm
        rows.append(
            {
                "frame": det.frame,
                "frame_index": det.frame_index,
                "track_id": det.track_id,
                "class_id": det.class_id,
                "class_name": det.class_name,
                "confidence": det.confidence,
                "xc_px": det.xc_px,
                "yc_px": det.yc_px,
                "area_px2": det.area_px2,
                "perimeter_px": det.perimeter_px,
                "circularity": det.circularity,
                "diameter_px": det.diameter_px,
                "xc_mm": det.xc_px * scale if scale else np.nan,
                "yc_mm": det.yc_px * scale if scale else np.nan,
                "area_mm2": det.area_px2 * scale * scale if scale else np.nan,
                "perimeter_mm": det.perimeter_px * scale if scale else np.nan,
                "diameter_mm": det.diameter_px * scale if scale else np.nan,
                "interpolated": False,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if interpolate_missing:
        df = add_interpolated_rows(df)

    if fit_tracks:
        df = add_fitted_track_columns(df, max_degree=track_fit_degree, px_size_mm=px_size_mm)
    else:
        df["xc_fit_px"] = np.nan
        df["yc_fit_px"] = np.nan
        df["fit_residual_px"] = np.nan
        df["fit_degree"] = np.nan
        df["fit_used"] = False
        df["xc_fit_mm"] = np.nan
        df["yc_fit_mm"] = np.nan

    df = df.sort_values(["track_id", "frame_index"]).reset_index(drop=True)
    dt = 1.0 / fps
    for col in [
        "vx_px_s",
        "vy_px_s",
        "speed_px_s",
        "vx_fit_px_s",
        "vy_fit_px_s",
        "speed_fit_px_s",
        "vx_mm_s",
        "vy_mm_s",
        "speed_mm_s",
        "vx_fit_mm_s",
        "vy_fit_mm_s",
        "speed_fit_mm_s",
    ]:
        df[col] = np.nan

    for _, group in df.groupby("track_id", sort=False):
        indices = group.index.to_list()
        if len(indices) < 2:
            continue
        for prev_idx, cur_idx in zip(indices[:-1], indices[1:]):
            frame_gap = int(df.at[cur_idx, "frame_index"] - df.at[prev_idx, "frame_index"])
            if frame_gap <= 0:
                continue
            actual_dt = dt * frame_gap
            vx = (df.at[cur_idx, "xc_px"] - df.at[prev_idx, "xc_px"]) / actual_dt
            vy = (df.at[cur_idx, "yc_px"] - df.at[prev_idx, "yc_px"]) / actual_dt
            speed = math.hypot(vx, vy)
            df.at[cur_idx, "vx_px_s"] = vx
            df.at[cur_idx, "vy_px_s"] = vy
            df.at[cur_idx, "speed_px_s"] = speed

            if use_fitted_for_speed and pd.notna(df.at[cur_idx, "xc_fit_px"]) and pd.notna(df.at[prev_idx, "xc_fit_px"]):
                fit_x_col = "xc_fit_px"
                fit_y_col = "yc_fit_px"
            else:
                fit_x_col = "xc_px"
                fit_y_col = "yc_px"

            vx_fit = (df.at[cur_idx, fit_x_col] - df.at[prev_idx, fit_x_col]) / actual_dt
            vy_fit = (df.at[cur_idx, fit_y_col] - df.at[prev_idx, fit_y_col]) / actual_dt
            speed_fit = math.hypot(vx_fit, vy_fit)
            df.at[cur_idx, "vx_fit_px_s"] = vx_fit
            df.at[cur_idx, "vy_fit_px_s"] = vy_fit
            df.at[cur_idx, "speed_fit_px_s"] = speed_fit
            if px_size_mm:
                df.at[cur_idx, "vx_mm_s"] = vx * px_size_mm
                df.at[cur_idx, "vy_mm_s"] = vy * px_size_mm
                df.at[cur_idx, "speed_mm_s"] = speed * px_size_mm
                df.at[cur_idx, "vx_fit_mm_s"] = vx_fit * px_size_mm
                df.at[cur_idx, "vy_fit_mm_s"] = vy_fit * px_size_mm
                df.at[cur_idx, "speed_fit_mm_s"] = speed_fit * px_size_mm

    return df.sort_values(["frame_index", "track_id"]).reset_index(drop=True)


def add_track_quality_columns(df: pd.DataFrame, min_real_points: int) -> pd.DataFrame:
    if df.empty or "track_id" not in df:
        return df
    df = df.copy()
    df["track_real_points"] = 0
    df["track_total_points"] = 0
    df["track_frame_span"] = 0
    df["track_quality"] = "unknown"

    for track_id, group in df.groupby("track_id", sort=False):
        real_group = group[~group["interpolated"].astype(bool)] if "interpolated" in group else group
        real_points = len(real_group)
        total_points = len(group)
        frame_span = int(group["frame_index"].max() - group["frame_index"].min() + 1) if len(group) else 0
        residual_max = group["fit_residual_px"].max(skipna=True) if "fit_residual_px" in group else np.nan
        speed_max = group["speed_fit_px_s"].max(skipna=True) if "speed_fit_px_s" in group else np.nan

        if real_points < min_real_points:
            quality = "short"
        elif pd.notna(residual_max) and residual_max > 80:
            quality = "high_residual"
        elif pd.notna(speed_max) and speed_max > 3500:
            quality = "high_speed"
        else:
            quality = "ok"

        mask = df["track_id"] == track_id
        df.loc[mask, "track_real_points"] = real_points
        df.loc[mask, "track_total_points"] = total_points
        df.loc[mask, "track_frame_span"] = frame_span
        df.loc[mask, "track_quality"] = quality

    return df


def build_track_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "track_id" not in df:
        return pd.DataFrame()

    rows = []
    for track_id, group in df.groupby("track_id", sort=True):
        real_group = group[~group["interpolated"].astype(bool)] if "interpolated" in group else group
        rows.append(
            {
                "track_id": track_id,
                "quality": group["track_quality"].iloc[0] if "track_quality" in group else "",
                "real_points": len(real_group),
                "total_points": len(group),
                "frame_start": int(group["frame_index"].min()),
                "frame_end": int(group["frame_index"].max()),
                "frame_span": int(group["frame_index"].max() - group["frame_index"].min() + 1),
                "mean_confidence": real_group["confidence"].mean(skipna=True) if "confidence" in real_group else np.nan,
                "max_fit_residual_px": group["fit_residual_px"].max(skipna=True) if "fit_residual_px" in group else np.nan,
                "max_speed_fit_px_s": group["speed_fit_px_s"].max(skipna=True) if "speed_fit_px_s" in group else np.nan,
                "median_speed_fit_px_s": group["speed_fit_px_s"].median(skipna=True) if "speed_fit_px_s" in group else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_fitted_track_columns(df: pd.DataFrame, max_degree: int, px_size_mm: float | None) -> pd.DataFrame:
    df = df.copy()
    for col in ["xc_fit_px", "yc_fit_px", "fit_residual_px", "fit_degree", "xc_fit_mm", "yc_fit_mm"]:
        df[col] = np.nan
    df["fit_used"] = False

    if df.empty or "track_id" not in df:
        return df

    for _, group in df.groupby("track_id", sort=False):
        group = group.sort_values("frame_index")
        real_group = group[~group["interpolated"].astype(bool)] if "interpolated" in group else group
        if len(real_group) < 2:
            continue

        degree = min(max_degree, len(real_group) - 1)
        if degree < 1:
            continue

        t_real = real_group["frame_index"].to_numpy(dtype=float)
        x_real = real_group["xc_px"].to_numpy(dtype=float)
        y_real = real_group["yc_px"].to_numpy(dtype=float)
        t0 = float(t_real.min())
        t_real_centered = t_real - t0

        try:
            x_coef = np.polyfit(t_real_centered, x_real, degree)
            y_coef = np.polyfit(t_real_centered, y_real, degree)
        except np.linalg.LinAlgError:
            if degree == 1:
                continue
            degree = 1
            x_coef = np.polyfit(t_real_centered, x_real, degree)
            y_coef = np.polyfit(t_real_centered, y_real, degree)

        indices = group.index.to_list()
        t_all = df.loc[indices, "frame_index"].to_numpy(dtype=float) - t0
        x_fit = np.polyval(x_coef, t_all)
        y_fit = np.polyval(y_coef, t_all)
        residual = np.sqrt((df.loc[indices, "xc_px"].to_numpy(dtype=float) - x_fit) ** 2 + (df.loc[indices, "yc_px"].to_numpy(dtype=float) - y_fit) ** 2)

        df.loc[indices, "xc_fit_px"] = x_fit
        df.loc[indices, "yc_fit_px"] = y_fit
        df.loc[indices, "fit_residual_px"] = residual
        df.loc[indices, "fit_degree"] = degree
        df.loc[indices, "fit_used"] = True
        if px_size_mm:
            df.loc[indices, "xc_fit_mm"] = x_fit * px_size_mm
            df.loc[indices, "yc_fit_mm"] = y_fit * px_size_mm

    return df


def add_interpolated_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = [row.to_dict() for _, row in df.iterrows()]
    numeric_cols = [
        "confidence",
        "xc_px",
        "yc_px",
        "area_px2",
        "perimeter_px",
        "circularity",
        "diameter_px",
        "xc_mm",
        "yc_mm",
        "area_mm2",
        "perimeter_mm",
        "diameter_mm",
    ]

    for _, group in df.sort_values(["track_id", "frame_index"]).groupby("track_id", sort=False):
        group = group.sort_values("frame_index")
        records = group.to_dict("records")
        for prev, cur in zip(records[:-1], records[1:]):
            frame_gap = int(cur["frame_index"] - prev["frame_index"])
            if frame_gap <= 1:
                continue
            for step in range(1, frame_gap):
                ratio = step / frame_gap
                interpolated = dict(prev)
                interpolated["frame_index"] = int(prev["frame_index"] + step)
                interpolated["frame"] = f"interpolated_{int(prev['frame_index'])}_{int(cur['frame_index'])}_{step}"
                interpolated["confidence"] = np.nan
                interpolated["interpolated"] = True
                for col in numeric_cols:
                    if col == "confidence":
                        continue
                    prev_value = prev.get(col, np.nan)
                    cur_value = cur.get(col, np.nan)
                    if pd.isna(prev_value) or pd.isna(cur_value):
                        interpolated[col] = np.nan
                    else:
                        interpolated[col] = float(prev_value) + (float(cur_value) - float(prev_value)) * ratio
                rows.append(interpolated)

    return pd.DataFrame(rows)


def draw_overlay(
    image_paths: list[Path],
    detections: list[Detection],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_frame: dict[str, list[Detection]] = {}
    for det in detections:
        by_frame.setdefault(det.frame, []).append(det)

    for image_path in image_paths:
        image = read_image(image_path)
        if image is None:
            continue

        overlay = image.copy()
        for det in by_frame.get(image_path.name, []):
            contour_i = det.contour.astype(np.int32)
            cv2.drawContours(overlay, [contour_i], -1, (0, 255, 255), thickness=-1)
            cv2.drawContours(image, [contour_i], -1, (0, 180, 255), thickness=2)
            label = f"ID {det.track_id} cls{det.class_id} C={det.circularity:.2f}"
            cv2.circle(image, (round(det.xc_px), round(det.yc_px)), 3, (255, 0, 0), -1)
            cv2.putText(
                image,
                label,
                (round(det.xc_px) + 6, round(det.yc_px) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        image = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)
        write_image(output_dir / image_path.name, image)


def track_real_point_count(group: pd.DataFrame) -> int:
    if "interpolated" in group.columns:
        return int((~group["interpolated"].astype(bool)).sum())
    return len(group)


def draw_tracks(df: pd.DataFrame, image_paths: list[Path], output_path: Path) -> None:
    if df.empty or not image_paths:
        return

    first = read_image(image_paths[0])
    if first is None:
        return

    canvas = first.copy()
    for track_id, group in df.groupby("track_id"):
        if track_real_point_count(group) < 2:
            continue
        group = group.sort_values("frame_index")
        points = group[["xc_px", "yc_px"]].to_numpy(dtype=np.float32)
        if len(points) == 0:
            continue
        color = (
            int((37 * int(track_id)) % 255),
            int((97 * int(track_id)) % 255),
            int((173 * int(track_id)) % 255),
        )
        for p1, p2 in zip(points[:-1], points[1:]):
            cv2.line(canvas, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), color, 2)

        if {"xc_fit_px", "yc_fit_px"}.issubset(group.columns) and group[["xc_fit_px", "yc_fit_px"]].notna().all(axis=1).sum() >= 2:
            fit_points = group[["xc_fit_px", "yc_fit_px"]].dropna().to_numpy(dtype=np.float32)
            for p1, p2 in zip(fit_points[:-1], fit_points[1:]):
                cv2.line(canvas, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), (255, 255, 255), 1)

        start = tuple(np.round(points[0]).astype(int))
        cv2.circle(canvas, start, 4, color, -1)
        cv2.putText(canvas, f"ID {track_id}", (start[0] + 6, start[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_image(output_path, canvas)


def _open_video_writer(
    path: Path,
    fps: float,
    width: int,
    height: int,
) -> tuple[cv2.VideoWriter, Path]:
    attempts = [
        (path, cv2.VideoWriter_fourcc(*"mp4v")),
        (path.with_suffix(".avi"), cv2.VideoWriter_fourcc(*"MJPG")),
    ]
    for candidate, fourcc in attempts:
        writer = cv2.VideoWriter(str(candidate), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer, candidate
        writer.release()
    raise RuntimeError(f"Could not open a video writer for: {path}")


def draw_track_videos(
    df: pd.DataFrame,
    image_paths: list[Path],
    output_dir: Path,
    fps: float,
    slowdown: float,
) -> list[Path]:
    """Render one short video per track, ordered by start time.

    Each clip shows a single particle's trajectory growing frame by frame:
    filled circles mark real detections, hollow circles mark interpolated
    positions, and the trailing line shows the accumulated path. Tracks with
    fewer than two real points are skipped because they carry no motion.

    ``slowdown`` is how many times slower than real time the clips play back.
    The video frame rate becomes ``fps / slowdown``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if df.empty or not image_paths:
        return []

    first = read_image(image_paths[0])
    if first is None:
        return []
    height, width = first.shape[:2]

    if slowdown <= 0:
        slowdown = 1.0
    video_fps = fps / slowdown

    tracks: list[tuple[int, int, pd.DataFrame]] = []
    for track_id, group in df.groupby("track_id", sort=False):
        if track_real_point_count(group) < 2:
            continue
        tracks.append((int(group["frame_index"].min()), int(track_id), group))
    tracks.sort(key=lambda item: (item[0], item[1]))

    written: list[Path] = []
    info_rows: list[dict[str, object]] = []
    for seq, (_, track_id, group) in enumerate(tracks, start=1):
        group = group.sort_values("frame_index")
        by_frame = {int(row["frame_index"]): row for _, row in group.iterrows()}
        color = (
            int((37 * int(track_id)) % 255),
            int((97 * int(track_id)) % 255),
            int((173 * int(track_id)) % 255),
        )

        video_path = output_dir / f"{seq:02d}_ID{track_id}.mp4"
        writer, video_path = _open_video_writer(video_path, video_fps, width, height)

        trail: list[tuple[float, float, bool]] = []
        frame_start = int(group["frame_index"].min())
        frame_end = int(group["frame_index"].max())
        video_frames = 0
        for frame_index in range(frame_start, frame_end + 1):
            if 0 <= frame_index < len(image_paths):
                frame = read_image(image_paths[frame_index])
                frame = frame.copy() if frame is not None else np.full((height, width, 3), 40, dtype=np.uint8)
            else:
                frame = np.full((height, width, 3), 40, dtype=np.uint8)

            row = by_frame.get(frame_index)
            if row is not None:
                trail.append((float(row["xc_px"]), float(row["yc_px"]), not bool(row.get("interpolated", False))))

            for (x1, y1, _), (x2, y2, _) in zip(trail[:-1], trail[1:]):
                cv2.line(frame, (round(x1), round(y1)), (round(x2), round(y2)), color, 2)
            for x, y, is_real in trail:
                cv2.circle(frame, (round(x), round(y)), 3, color, -1 if is_real else 1)
            if trail:
                cx, cy, _ = trail[-1]
                cv2.putText(frame, f"ID {track_id}", (round(cx) + 8, round(cy) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            writer.write(frame)
            video_frames += 1

        writer.release()
        written.append(video_path)
        info_rows.append(
            {
                "seq": seq,
                "file": video_path.name,
                "track_id": track_id,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "video_frames": video_frames,
                "duration_s": video_frames / video_fps,
            }
        )

    _write_video_info(output_dir, fps, slowdown, video_fps, info_rows)
    return written


def _write_video_info(
    output_dir: Path,
    fps: float,
    slowdown: float,
    video_fps: float,
    info_rows: list[dict[str, object]],
) -> None:
    lines = [
        "轨迹视频参数",
        "============",
        "",
        "[全局]",
        f"源帧率 fps      : {fps:g}",
        f"慢放倍数        : {slowdown:g}",
        f"视频帧率        : {video_fps:.2f} fps",
        f"视频数          : {len(info_rows)}",
        f"输出目录        : {output_dir}",
        "",
        "[各视频] 按起始时间排序",
        "序号  文件          轨迹ID  起始帧  结束帧  视频帧数  时长(s)",
        "----------------------------------------------------------------",
    ]
    for row in info_rows:
        lines.append(
            f"{row['seq']:>2}    {str(row['file']):<14} {int(row['track_id']):>6} "
            f"{int(row['frame_start']):>6} {int(row['frame_end']):>6} "
            f"{int(row['video_frames']):>8} {float(row['duration_s']):>8.2f}"
        )
    (output_dir / "video_info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def draw_full_video(
    df: pd.DataFrame,
    image_paths: list[Path],
    output_path: Path,
    fps: float,
    slowdown: float,
) -> Path | None:
    """Render the whole sequence as one video with all trajectories overlaid.

    Plays every frame in order; each droplet's accumulated trail grows over time
    so the viewer sees all trajectories evolve together. Tracks with fewer than
    two real points are skipped.
    """
    if df.empty or not image_paths:
        return None

    first = read_image(image_paths[0])
    if first is None:
        return None
    height, width = first.shape[:2]

    if slowdown <= 0:
        slowdown = 1.0
    video_fps = fps / slowdown

    tracks: dict[int, pd.DataFrame] = {}
    for track_id, group in df.groupby("track_id", sort=False):
        if track_real_point_count(group) < 2:
            continue
        tracks[int(track_id)] = group.sort_values("frame_index")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer, output_path = _open_video_writer(output_path, video_fps, width, height)

    for frame_index in range(len(image_paths)):
        frame = read_image(image_paths[frame_index])
        frame = frame.copy() if frame is not None else np.full((height, width, 3), 40, dtype=np.uint8)

        for track_id, group in tracks.items():
            color = (
                int((37 * int(track_id)) % 255),
                int((97 * int(track_id)) % 255),
                int((173 * int(track_id)) % 255),
            )
            sub = group[group["frame_index"] <= frame_index]
            if sub.empty:
                continue
            points = sub[["xc_px", "yc_px"]].to_numpy(dtype=np.float32)
            for p1, p2 in zip(points[:-1], points[1:]):
                cv2.line(frame, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), color, 2)

            is_interp = sub["interpolated"].to_numpy(dtype=bool) if "interpolated" in sub.columns else np.zeros(len(sub), dtype=bool)
            for (x, y), interp in zip(points, is_interp):
                cv2.circle(frame, (round(x), round(y)), 3, color, -1 if not interp else 1)

            cx, cy = points[-1]
            cv2.putText(frame, f"ID {track_id}", (round(cx) + 8, round(cy) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    _write_full_video_info(output_path, fps, slowdown, video_fps, len(image_paths), len(tracks))
    return output_path


def _write_full_video_info(
    video_path: Path,
    fps: float,
    slowdown: float,
    video_fps: float,
    total_frames: int,
    track_count: int,
) -> None:
    duration = total_frames / video_fps if video_fps > 0 else 0.0
    info_path = video_path.with_name(video_path.stem + "_info.txt")
    lines = [
        "全序列视频参数",
        "==============",
        "",
        "[全局]",
        f"源帧率 fps      : {fps:g}",
        f"慢放倍数        : {slowdown:g}",
        f"视频帧率        : {video_fps:.2f} fps",
        f"总帧数          : {total_frames}",
        f"时长            : {duration:.2f} s",
        f"显示轨迹数      : {track_count}",
        f"输出文件        : {video_path.name}",
    ]
    info_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def load_config(config_path: Path) -> dict[str, object]:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain key-value pairs: {config_path}")
    return data


def config_value(config: dict[str, object], key: str, default: object = None) -> object:
    value = config.get(key, default)
    return default if value is None else value


def parse_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def make_timestamp_output_dir(base_output: Path, args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conf_tag = f"conf{str(args.conf).replace('.', '')}"
    imgsz_tag = f"img{args.imgsz}"
    tracker_tag = str(args.tracker)
    return base_output / f"run_{timestamp}_{conf_tag}_{imgsz_tag}_{tracker_tag}"


def make_numeric_output_dir(base_output: Path, version_step: float) -> Path:
    pattern = re.compile(r"^run(\d+(?:\.\d+)?)$")
    versions: list[float] = []
    if base_output.exists():
        for child in base_output.iterdir():
            if not child.is_dir():
                continue
            match = pattern.match(child.name)
            if match:
                versions.append(float(match.group(1)))

    next_version = (max(versions) + version_step) if versions else version_step
    version_text = f"{next_version:.1f}"
    candidate = base_output / f"run{version_text}"
    while candidate.exists():
        next_version += version_step
        version_text = f"{next_version:.1f}"
        candidate = base_output / f"run{version_text}"
    return candidate


def make_run_output_dir(base_output: Path, args: argparse.Namespace) -> Path:
    if args.output_version_mode == "numeric":
        return make_numeric_output_dir(base_output, args.output_version_step)
    return make_timestamp_output_dir(base_output, args)


def build_arg_parser(config: dict[str, object]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO-seg aluminum droplet analysis pipeline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML config file path.")
    parser.add_argument("--images", type=Path, default=Path(str(config_value(config, "images", "input/exp01"))), help="Folder containing continuous frames.")
    parser.add_argument("--model", type=Path, default=Path(str(config_value(config, "model", "weights/best.pt"))), help="YOLO-seg model path.")
    parser.add_argument("--output", type=Path, default=Path(str(config_value(config, "output", "output/exp01"))), help="Output folder.")
    parser.add_argument("--fps", type=float, default=float(config_value(config, "fps", 30.0)), help="Frame rate of the image sequence.")
    parser.add_argument("--px-size-mm", type=float, default=config.get("px_size_mm"), help="Physical size of one pixel in mm.")
    parser.add_argument("--target-class", type=str, default=config_value(config, "target_class", "飞溅液滴"), help="Class name or id to analyze. Use comma for multiple classes.")
    parser.add_argument("--conf", type=float, default=float(config_value(config, "conf", 0.25)), help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=int(config_value(config, "imgsz", 1024)), help="YOLO inference image size.")
    parser.add_argument("--device", type=str, default=config.get("device"), help="YOLO device, for example 0, cpu, cuda:0.")
    parser.add_argument("--tracker", choices=["bytetrack", "nearest", "none"], default=str(config_value(config, "tracker", "bytetrack")), help="Tracking method for assigning droplet IDs.")
    parser.add_argument("--tracker-config", type=str, default=str(config_value(config, "tracker_config", "bytetrack.yaml")), help="Ultralytics tracker config used when --tracker bytetrack.")
    parser.add_argument("--max-track-distance", type=float, default=float(config_value(config, "max_track_distance", 80.0)), help="Max center distance for nearest-neighbor tracking, in pixels.")
    parser.add_argument("--max-track-gap", type=int, default=int(config_value(config, "max_track_gap", 0)), help="Max missing frames allowed when linking broken tracklets.")
    parser.add_argument("--max-gap-link-distance", type=float, default=float(config_value(config, "max_gap_link_distance", 160.0)), help="Max prediction distance for linking broken tracklets, in pixels.")
    parser.add_argument("--max-size-ratio", type=float, default=float(config_value(config, "max_size_ratio", 3.0)), help="Max area ratio allowed when linking a detection to a track (>1, smaller is stricter).")
    parser.add_argument("--max-color-distance", type=float, default=float(config_value(config, "max_color_distance", 70.0)), help="Max mean mask color distance (BGR L2) allowed when linking a detection to a track.")
    parser.add_argument("--suppress-overlap", action=argparse.BooleanOptionalAction, default=parse_bool(config.get("suppress_overlap"), True), help="Remove near-duplicate/halo detections within each frame before tracking.")
    parser.add_argument("--suppress-iou", type=float, default=float(config_value(config, "suppress_iou", 0.20)), help="Mask IoU threshold for within-frame duplicate suppression.")
    parser.add_argument("--suppress-center-ratio", type=float, default=float(config_value(config, "suppress_center_ratio", 1.0)), help="Center distance (as a ratio of the larger diameter) for halo/duplicate suppression.")
    parser.add_argument("--interpolate-missing", action=argparse.BooleanOptionalAction, default=parse_bool(config.get("interpolate_missing"), False), help="Add linearly interpolated rows for missing frames in each linked track.")
    parser.add_argument("--fit-tracks", action=argparse.BooleanOptionalAction, default=parse_bool(config.get("fit_tracks"), True), help="Fit each track with a low-order polynomial and output fitted coordinates.")
    parser.add_argument("--track-fit-degree", type=int, default=int(config_value(config, "track_fit_degree", 2)), help="Maximum polynomial degree for track fitting.")
    parser.add_argument("--use-fitted-for-speed", action=argparse.BooleanOptionalAction, default=parse_bool(config.get("use_fitted_for_speed"), True), help="Use fitted coordinates for smoothed speed columns.")
    parser.add_argument("--min-track-real-points", type=int, default=int(config_value(config, "min_track_real_points", 3)), help="Minimum real detections required for an ok-quality track.")
    parser.add_argument("--auto-output-subdir", action=argparse.BooleanOptionalAction, default=parse_bool(config.get("auto_output_subdir"), True), help="Create a timestamped subfolder under output for each run.")
    parser.add_argument("--output-version-mode", choices=["numeric", "timestamp"], default=str(config_value(config, "output_version_mode", "numeric")), help="Output subfolder naming mode when auto output subdir is enabled.")
    parser.add_argument("--output-version-step", type=float, default=float(config_value(config, "output_version_step", 1.0)), help="Version increment used by numeric output mode.")
    parser.add_argument("--save-track-videos", action=argparse.BooleanOptionalAction, default=parse_bool(config.get("save_track_videos"), True), help="Render one short video per track, ordered by start time.")
    parser.add_argument("--video-slowdown", type=float, default=float(config_value(config, "video_slowdown", 5.0)), help="How many times slower than real time the track videos play back.")
    parser.add_argument("--save-full-video", action=argparse.BooleanOptionalAction, default=parse_bool(config.get("save_full_video"), True), help="Render the whole sequence as one video with all trajectories overlaid.")
    return parser


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config)
    args = build_arg_parser(config).parse_args()
    if args.auto_output_subdir:
        args.output = make_run_output_dir(args.output, args)

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.images.exists():
        raise FileNotFoundError(f"Image folder not found: {args.images}")

    image_paths = list_images(args.images)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {args.images}")

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    target_class_ids = parse_target_classes(args.target_class, model.names)
    print(f"Model: {args.model}")
    print(f"Model task: {model.task}")
    print(f"Model classes: {normalize_names(model.names)}")
    print(f"Target class filter: {args.target_class or 'ALL'}")
    print(f"conf={args.conf}, imgsz={args.imgsz}, tracker={args.tracker}")

    detections = run_inference(
        model=model,
        image_paths=image_paths,
        target_class_ids=target_class_ids,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )
    raw_detection_count = len(detections)
    if args.suppress_overlap:
        detections = suppress_overlapping_detections(
            detections,
            iou_threshold=args.suppress_iou,
            center_ratio=args.suppress_center_ratio,
        )
    if args.tracker == "nearest":
        assign_tracks_motion(
            detections,
            max_distance_px=args.max_track_distance,
            max_gap=args.max_track_gap,
            max_size_ratio=args.max_size_ratio,
            max_color_distance=args.max_color_distance,
        )
    elif args.tracker == "bytetrack":
        print("Warning: bytetrack mode is deprecated in this project because it can change visible detections. Using motion-based tracking on predict() results instead.")
        assign_tracks_motion(
            detections,
            max_distance_px=args.max_track_distance,
            max_gap=args.max_track_gap,
            max_size_ratio=args.max_size_ratio,
            max_color_distance=args.max_color_distance,
        )

    links_made = link_tracks_across_gaps(
        detections,
        max_gap=args.max_track_gap,
        max_link_distance_px=args.max_gap_link_distance,
        max_size_ratio=args.max_size_ratio,
        max_color_distance=args.max_color_distance,
    )

    df = detections_to_dataframe(
        detections,
        fps=args.fps,
        px_size_mm=args.px_size_mm,
        interpolate_missing=args.interpolate_missing,
        fit_tracks=args.fit_tracks,
        track_fit_degree=args.track_fit_degree,
        use_fitted_for_speed=args.use_fitted_for_speed,
    )
    df = add_track_quality_columns(df, min_real_points=args.min_track_real_points)
    class_counts: dict[str, int] = {}
    for det in detections:
        class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
    csv_path = args.output / "result.csv"
    xlsx_path = args.output / "result.xlsx"
    summary_path = args.output / "summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_excel(xlsx_path, index=False)
    build_track_summary(df).to_csv(summary_path, index=False, encoding="utf-8-sig")

    draw_overlay(image_paths, detections, args.output / "overlay")
    draw_tracks(df, image_paths, args.output / "tracks.png")

    if args.save_track_videos:
        video_paths = draw_track_videos(df, image_paths, args.output / "tracks_videos", args.fps, args.video_slowdown)

    if args.save_full_video:
        full_video_path = draw_full_video(df, image_paths, args.output / "full_sequence.mp4", args.fps, args.video_slowdown)

    print(f"Processed images: {len(image_paths)}")
    if args.suppress_overlap:
        print(f"Suppressed overlapping/halo detections: {raw_detection_count - len(detections)}")
    print(f"Detections: {len(detections)}")
    print(f"Detections by class: {class_counts}")
    print(f"Track links across gaps: {links_made}")
    print(f"Interpolated missing frames: {args.interpolate_missing}")
    print(f"Track fitting: {args.fit_tracks}, degree={args.track_fit_degree}, fitted speed={args.use_fitted_for_speed}")
    if not df.empty and "track_quality" in df:
        print(f"Track quality counts: {df.drop_duplicates('track_id')['track_quality'].value_counts().to_dict()}")
    print(f"CSV: {csv_path}")
    print(f"Excel: {xlsx_path}")
    print(f"Summary: {summary_path}")
    print(f"Overlay images: {args.output / 'overlay'}")
    print(f"Tracks image: {args.output / 'tracks.png'}")
    if args.save_track_videos:
        print(f"Track videos: {len(video_paths)} in {args.output / 'tracks_videos'}")
    if args.save_full_video:
        print(f"Full sequence video: {full_video_path}")


if __name__ == "__main__":
    main()
