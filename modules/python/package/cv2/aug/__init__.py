"""Python augmentation composition helpers.

This module provides callables compatible with Compose/OneOf/Sometimes style
pipelines and carries explicit capability metadata.
"""

from __future__ import annotations

import random


class TransformCapability(object):
    def __init__(self, supported_targets=("image",), may_mutate_input=False):
        self.supported_targets = tuple(supported_targets)
        self.may_mutate_input = bool(may_mutate_input)


class TransformBase(object):
    def __call__(self, data):
        return self.apply(data)

    def apply(self, data):
        raise NotImplementedError()

    @property
    def capability(self):
        return TransformCapability()


class LambdaTransform(TransformBase):
    def __init__(self, fn, supported_targets=("image",), may_mutate_input=False, name=None):
        if fn is None:
            raise ValueError("fn must be provided")
        self._fn = fn
        self._capability = TransformCapability(supported_targets, may_mutate_input)
        self.name = name or getattr(fn, "__name__", "lambda_transform")

    @property
    def capability(self):
        return self._capability

    def apply(self, data):
        return self._fn(data)


class Compose(TransformBase):
    def __init__(self, transforms):
        self.transforms = [ensure_transform(t) for t in transforms]

    @property
    def capability(self):
        if not self.transforms:
            return TransformCapability()
        supported = set(self.transforms[0].capability.supported_targets)
        may_mutate = False
        for tr in self.transforms:
            supported &= set(tr.capability.supported_targets)
            may_mutate = may_mutate or tr.capability.may_mutate_input
        return TransformCapability(tuple(sorted(supported)), may_mutate)

    def apply(self, data):
        out = data
        for tr in self.transforms:
            out = tr(out)
        return out


class OneOf(TransformBase):
    def __init__(self, transforms, p=None):
        self.transforms = [ensure_transform(t) for t in transforms]
        if not self.transforms:
            raise ValueError("OneOf requires at least one transform")
        if p is None:
            self.p = [1.0 / len(self.transforms)] * len(self.transforms)
        else:
            if len(p) != len(self.transforms):
                raise ValueError("Probability list size mismatch")
            s = float(sum(p))
            if s <= 0.0:
                raise ValueError("Probability sum must be positive")
            self.p = [float(v) / s for v in p]

    @property
    def capability(self):
        supported = set(self.transforms[0].capability.supported_targets)
        may_mutate = False
        for tr in self.transforms:
            supported &= set(tr.capability.supported_targets)
            may_mutate = may_mutate or tr.capability.may_mutate_input
        return TransformCapability(tuple(sorted(supported)), may_mutate)

    def apply(self, data):
        tr = random.choices(self.transforms, weights=self.p, k=1)[0]
        return tr(data)


class Sometimes(TransformBase):
    def __init__(self, transform, probability=0.5):
        if probability < 0.0 or probability > 1.0:
            raise ValueError("probability must be in [0,1]")
        self.transform = ensure_transform(transform)
        self.probability = float(probability)

    @property
    def capability(self):
        return self.transform.capability

    def apply(self, data):
        if random.random() < self.probability:
            return self.transform(data)
        return data


def ensure_transform(obj):
    if isinstance(obj, TransformBase):
        return obj
    if callable(obj):
        return LambdaTransform(obj)
    raise TypeError("Transform must be callable or TransformBase")
