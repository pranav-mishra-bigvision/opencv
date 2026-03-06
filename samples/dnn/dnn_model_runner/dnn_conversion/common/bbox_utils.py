from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


VOC = "voc"
COCO = "coco"
YOLO = "yolo"
SUPPORTED_FORMATS = (VOC, COCO, YOLO)


@dataclass(frozen=True)
class BBoxPolicy:
    clip: bool = True
    min_area: float = 0.0
    min_visibility: float = 0.0
    drop_invalid: bool = True


@dataclass(frozen=True)
class ProcessedBBox:
    bbox: List[float]
    visibility: float
    area: float


def _validate_format(fmt: str) -> None:
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError("Unsupported bbox format '{}'. Supported formats: {}".format(fmt, SUPPORTED_FORMATS))


def _as_float_list(bbox: Sequence[float]) -> List[float]:
    if len(bbox) != 4:
        raise ValueError("Bounding box must have 4 elements, got {}".format(len(bbox)))
    return [float(v) for v in bbox]


def xyxy_area(bbox_xyxy: Sequence[float]) -> float:
    x1, y1, x2, y2 = _as_float_list(bbox_xyxy)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def clip_xyxy(bbox_xyxy: Sequence[float], image_size: Tuple[float, float]) -> List[float]:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size values must be positive")

    x1, y1, x2, y2 = _as_float_list(bbox_xyxy)
    return [
        min(max(x1, 0.0), width),
        min(max(y1, 0.0), height),
        min(max(x2, 0.0), width),
        min(max(y2, 0.0), height),
    ]


def to_xyxy(bbox: Sequence[float], fmt: str, image_size: Tuple[float, float], yolo_normalized: bool = True) -> List[float]:
    _validate_format(fmt)
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size values must be positive")

    b = _as_float_list(bbox)

    if fmt == VOC:
        return b

    if fmt == COCO:
        x, y, w, h = b
        return [x, y, x + w, y + h]

    cx, cy, w, h = b
    if yolo_normalized:
        cx *= width
        cy *= height
        w *= width
        h *= height
    half_w = w / 2.0
    half_h = h / 2.0
    return [cx - half_w, cy - half_h, cx + half_w, cy + half_h]


def from_xyxy(bbox_xyxy: Sequence[float], fmt: str, image_size: Tuple[float, float], yolo_normalized: bool = True) -> List[float]:
    _validate_format(fmt)
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size values must be positive")

    x1, y1, x2, y2 = _as_float_list(bbox_xyxy)

    if fmt == VOC:
        return [x1, y1, x2, y2]

    if fmt == COCO:
        return [x1, y1, x2 - x1, y2 - y1]

    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0

    if yolo_normalized:
        return [cx / width, cy / height, w / width, h / height]
    return [cx, cy, w, h]


def convert_bbox(
    bbox: Sequence[float],
    src_format: str,
    dst_format: str,
    image_size: Tuple[float, float],
    *,
    policy: Optional[BBoxPolicy] = None,
    yolo_normalized: bool = True,
) -> Optional[ProcessedBBox]:
    _validate_format(src_format)
    _validate_format(dst_format)

    current_policy = policy or BBoxPolicy()
    bbox_xyxy = to_xyxy(bbox, src_format, image_size, yolo_normalized=yolo_normalized)

    source_area = xyxy_area(bbox_xyxy)
    if current_policy.drop_invalid and source_area <= 0.0:
        return None

    processed_xyxy = clip_xyxy(bbox_xyxy, image_size) if current_policy.clip else bbox_xyxy
    processed_area = xyxy_area(processed_xyxy)

    if current_policy.drop_invalid and processed_area <= 0.0:
        return None

    visibility = 0.0 if source_area <= 0.0 else processed_area / source_area

    if processed_area < current_policy.min_area:
        return None
    if visibility < current_policy.min_visibility:
        return None

    return ProcessedBBox(
        bbox=from_xyxy(processed_xyxy, dst_format, image_size, yolo_normalized=yolo_normalized),
        visibility=visibility,
        area=processed_area,
    )


def convert_bboxes(
    bboxes: Iterable[Sequence[float]],
    src_format: str,
    dst_format: str,
    image_size: Tuple[float, float],
    *,
    policy: Optional[BBoxPolicy] = None,
    yolo_normalized: bool = True,
) -> List[ProcessedBBox]:
    results: List[ProcessedBBox] = []
    for bbox in bboxes:
        converted = convert_bbox(
            bbox,
            src_format,
            dst_format,
            image_size,
            policy=policy,
            yolo_normalized=yolo_normalized,
        )
        if converted is not None:
            results.append(converted)
    return results
