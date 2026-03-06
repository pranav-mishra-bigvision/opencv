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
