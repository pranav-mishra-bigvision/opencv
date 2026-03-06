#!/usr/bin/env python

import random

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests


class AugmentationPythonHooks(NewOpenCVTests):

    def test_lambda_compose_sometimes_oneof(self):
        random.seed(0)

        add1 = cv.aug.LambdaTransform(lambda x: x + 1, supported_targets=("image", "mask"), may_mutate_input=False)
        mul2 = cv.aug.LambdaTransform(lambda x: x * 2, supported_targets=("image",), may_mutate_input=True)

        composed = cv.aug.Compose([add1, mul2])
        out = composed(3)
        self.assertEqual(out, 8)

        self.assertEqual(composed.capability.supported_targets, ("image",))
        self.assertTrue(composed.capability.may_mutate_input)

        sometimes = cv.aug.Sometimes(lambda x: x + 10, probability=0.0)
        self.assertEqual(sometimes(5), 5)

        one = cv.aug.OneOf([
            cv.aug.LambdaTransform(lambda x: x + 100),
            cv.aug.LambdaTransform(lambda x: x + 200),
        ], p=[1.0, 0.0])
        self.assertEqual(one(1), 101)


    def test_to_tensor_layout_shape_dtype_and_values(self):
        image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

        nchw = cv.aug.to_tensor_layout(image, layout="NCHW")
        self.assertEqual(nchw.shape, (1, 3, 2, 3))
        self.assertEqual(nchw.dtype, np.float32)
        self.assertTrue(np.array_equal(nchw[0, :, 0, 0], image[0, 0, :].astype(np.float32)))

        nhwc = cv.aug.to_tensor_layout(image, layout="NHWC", dtype=np.uint8)
        self.assertEqual(nhwc.shape, (1, 2, 3, 3))
        self.assertEqual(nhwc.dtype, np.uint8)
        self.assertTrue(np.array_equal(nhwc[0], image))

        normalized = cv.aug.to_tensor_layout(
            image,
            layout="NCHW",
            normalize=True,
            scale=1.0 / 255.0,
            mean=(0.1, 0.2, 0.3),
            std=(0.5, 0.25, 0.1),
        )
        expected = (image.astype(np.float32) / 255.0 - np.array([0.1, 0.2, 0.3], dtype=np.float32)) / np.array([0.5, 0.25, 0.1], dtype=np.float32)
        expected = np.transpose(expected[np.newaxis, ...], (0, 3, 1, 2))
        np.testing.assert_allclose(normalized, expected, rtol=0, atol=1e-6)

    def test_to_tensor_layout_memory_behavior(self):
        image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)

        nhwc_view = cv.aug.to_tensor_layout(image, layout="NHWC", dtype=np.uint8)
        self.assertTrue(np.shares_memory(nhwc_view[0], image))

        nhwc_copied = cv.aug.to_tensor_layout(image, layout="NHWC", dtype=np.uint8, copy=True)
        self.assertFalse(np.shares_memory(nhwc_copied[0], image))

        nhwc_normalized = cv.aug.to_tensor_layout(image, layout="NHWC", normalize=True)
        self.assertFalse(np.shares_memory(nhwc_normalized[0], image))

    def test_numpy_input_passthrough(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        pipeline = cv.aug.Compose([
            lambda x: x + 1,
            cv.aug.Sometimes(lambda x: x, probability=1.0),
        ])
        out = pipeline(arr)
        self.assertTrue(np.array_equal(out, arr + 1))


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
