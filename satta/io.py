"""Validated, portable JSON inputs. No pickle deserialization or model loading."""
import json
from pathlib import Path

import numpy as np

from .metrics import _prediction_array, gt_array


def read_json(path):
    # Accept UTF-8 JSON from both Windows PowerShell versions.
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def load_gt(path):
    data = read_json(path)
    if not isinstance(data, dict) or not data:
        raise ValueError("ground truth must be a nonempty image-name mapping")
    for image, record in data.items():
        if not isinstance(image, str) or not image or not isinstance(record, dict):
            raise ValueError("each ground-truth image needs a name and object record")
        boxes = np.asarray(record.get("boxes"), dtype=np.float32)
        if boxes.size and (boxes.ndim != 2 or boxes.shape[1] != 5):
            raise ValueError(f"{image}: GT boxes must be Nx5 (xyxy, class)")
        if "boxes" not in record:
            raise ValueError(f"{image}: missing GT boxes")
        gt_array(record)
        for field in ("width", "height"):
            if field in record and (not np.isfinite(record[field]) or record[field] <= 0):
                raise ValueError(f"{image}: {field} must be positive and finite")
    return data


def load_predictions(path, gts):
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError("predictions must be a JSON list of detection objects")
    by_image = {name: [] for name in gts}
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"prediction {index}: expected an object")
        name = row.get("image")
        if name not in gts:
            raise ValueError(f"prediction {index}: unknown image {name!r}")
        box = np.asarray(row.get("bbox"), dtype=np.float64)
        if box.shape != (4,):
            raise ValueError(f"prediction {index}: bbox must contain x, y, width, height")
        x, y, width, height = box
        if width < 0 or height < 0:
            raise ValueError(f"prediction {index}: bbox width and height cannot be negative")
        by_image[name].append([x, y, x + width, y + height, row["score"], row["category_id"]])
    return {name: _prediction_array(np.asarray(rows, np.float32).reshape(-1, 6))
            for name, rows in by_image.items()}


def load_candidates(path, gts, required_conf):
    data = read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("preds"), dict):
        raise ValueError("candidate JSON requires meta and preds objects")
    meta, records = data.get("meta", {}), data["preds"]
    if set(records) != set(gts):
        raise ValueError("candidate image names must exactly cover the GT image universe")
    if meta.get("complete") is not True:
        raise ValueError("candidate meta.complete must be true")
    floor = meta.get("conf")
    if not isinstance(floor, (float, int)) or not np.isfinite(floor) or floor < 0:
        raise ValueError("candidate meta.conf must record a nonnegative finite inference score floor")
    if floor > required_conf + 1e-12:
        raise ValueError("requested confidence is below the saved candidate score floor")
    result = {}
    for name, record in records.items():
        if not isinstance(record, dict) or not all(k in record for k in ("g", "t", "tl")):
            raise ValueError(f"{name}: candidate record requires g, t and tl")
        global_boxes, tiles = _prediction_array(record["g"]), _prediction_array(record["t"])
        centers = np.asarray(record["tl"], dtype=np.float64)
        if centers.size == 0:
            centers = centers.reshape(0, 2)
        if centers.shape != (len(tiles), 2) or not np.isfinite(centers).all():
            raise ValueError(f"{name}: tl must be a finite Nx2 array aligned with t")
        if np.any((centers < 0) | (centers > 1)):
            raise ValueError(f"{name}: normalized in-tile centers must lie in [0, 1]")
        result[name] = {"g": global_boxes, "t": tiles, "tl": centers}
    return meta, result


def json_safe(value):
    """Use JSON null for undefined scale-band metrics, not nonstandard NaN."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value
