import math
import unittest

from pathlib import Path
import sys

COMMON_DIR = Path(__file__).resolve().parents[1]
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from bbox_utils import (
    BBoxPolicy,
    COCO,
    VOC,
    YOLO,
    convert_bbox,
    convert_bboxes,
)


class TestBBoxUtils(unittest.TestCase):
    def test_conversion_across_voc_coco_yolo(self):
        image_size = (200, 100)
        voc = [10, 20, 110, 70]

        coco_box = convert_bbox(voc, VOC, COCO, image_size)
        self.assertIsNotNone(coco_box)
        self.assertEqual(coco_box.bbox, [10.0, 20.0, 100.0, 50.0])

        yolo_box = convert_bbox(coco_box.bbox, COCO, YOLO, image_size)
        self.assertIsNotNone(yolo_box)
        cx, cy, w, h = yolo_box.bbox
        self.assertAlmostEqual(cx, 0.3)
        self.assertAlmostEqual(cy, 0.45)
        self.assertAlmostEqual(w, 0.5)
        self.assertAlmostEqual(h, 0.5)

        voc_roundtrip = convert_bbox(yolo_box.bbox, YOLO, VOC, image_size)
        self.assertIsNotNone(voc_roundtrip)
        for expected, actual in zip(voc, voc_roundtrip.bbox):
            self.assertAlmostEqual(expected, actual)

    def test_negative_coords_are_clipped_with_visibility(self):
        image_size = (100, 100)
        src = [-10, -10, 20, 20]

        result = convert_bbox(src, VOC, VOC, image_size, policy=BBoxPolicy(clip=True, min_visibility=0.25))
        self.assertIsNotNone(result)
        self.assertEqual(result.bbox, [0.0, 0.0, 20.0, 20.0])
        self.assertAlmostEqual(result.visibility, 4.0 / 9.0)

        dropped = convert_bbox(src, VOC, VOC, image_size, policy=BBoxPolicy(clip=True, min_visibility=0.5))
        self.assertIsNone(dropped)

    def test_degenerate_boxes_dropped_by_default(self):
        image_size = (100, 100)

        degenerate_voc = [10, 10, 10, 20]
        self.assertIsNone(convert_bbox(degenerate_voc, VOC, COCO, image_size))

        degenerate_coco = [5, 5, 0, 10]
        self.assertIsNone(convert_bbox(degenerate_coco, COCO, VOC, image_size))

    def test_partial_out_of_frame_min_area_policy(self):
        image_size = (100, 100)
        src = [80, 80, 50, 50]  # coco box => visible is 20x20

        kept = convert_bbox(src, COCO, COCO, image_size, policy=BBoxPolicy(min_area=300))
        self.assertIsNotNone(kept)
        self.assertEqual(kept.bbox, [80.0, 80.0, 20.0, 20.0])
        self.assertAlmostEqual(kept.area, 400.0)

        dropped = convert_bbox(src, COCO, COCO, image_size, policy=BBoxPolicy(min_area=500))
        self.assertIsNone(dropped)

    def test_drop_invalid_false_keeps_degenerate(self):
        image_size = (100, 100)
        src = [0.5, 0.5, 0.0, 0.1]  # yolo width is zero

        result = convert_bbox(src, YOLO, COCO, image_size, policy=BBoxPolicy(drop_invalid=False))
        self.assertIsNotNone(result)
        self.assertEqual(result.bbox[2], 0.0)
        self.assertTrue(math.isclose(result.area, 0.0))

    def test_batch_conversion_drops_invalid(self):
        image_size = (100, 100)
        src_boxes = [
            [-10, -10, 20, 20],
            [10, 10, 10, 20],
            [95, 95, 105, 105],
        ]
        converted = convert_bboxes(
            src_boxes,
            VOC,
            VOC,
            image_size,
            policy=BBoxPolicy(min_visibility=0.1, min_area=1),
        )
        self.assertEqual(len(converted), 2)


if __name__ == "__main__":
    unittest.main()
