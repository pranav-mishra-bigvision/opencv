"""Python augmentation composition helpers.

This module provides callables compatible with Compose/OneOf/Sometimes style
pipelines and carries explicit capability metadata.
"""

from __future__ import annotations

import random

import numpy as np


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


def _ensure_hwc_or_nhwc(array):
    if array.ndim == 2:
        return array[..., np.newaxis], False
    if array.ndim == 3:
        return array, False
    if array.ndim == 4:
        return array, True
    raise ValueError("Expected 2D/3D/4D augmentation output, got shape {}".format(array.shape))


def to_tensor_layout(data, layout="NCHW", dtype=np.float32, normalize=False,
                     scale=1.0 / 255.0, mean=None, std=None, copy=False):
    """Convert augmentation outputs into tensor-ready image layouts.

    Parameters
    ----------
    data : numpy.ndarray
        Input image-like array in OpenCV's default channel convention (BGR for
        multi-channel images). Supported input shapes are ``H x W``, ``H x W x C``
        and ``N x H x W x C``.
    layout : {"NCHW", "NHWC"}
        Target tensor layout. ``NCHW`` is common for PyTorch/OpenCV-DNN and
        ``NHWC`` is common for TensorFlow runtimes.
    dtype : numpy dtype
        Output data type. Defaults to ``np.float32``.
    normalize : bool
        If ``True``, apply pixel normalization as ``(x * scale - mean) / std``.
        This function does not reorder channels; if your model expects RGB,
        convert BGR->RGB before calling.
    scale : float
        Multiplicative factor used when ``normalize=True``.
    mean : scalar or sequence, optional
        Mean subtraction value(s) used when ``normalize=True``.
    std : scalar or sequence, optional
        Standard deviation divisor value(s) used when ``normalize=True``.
    copy : bool
        Force a deep copy before conversion. When ``False``, this helper prefers
        zero-copy/view operations (reshape/transpose/astype(copy=False)).

    Returns
    -------
    numpy.ndarray
        Tensor in the requested layout. For non-batched inputs this function
        adds a batch dimension.
    """
    tensor = np.array(data, copy=copy)
    tensor, already_batched = _ensure_hwc_or_nhwc(tensor)

    if not already_batched:
        tensor = tensor[np.newaxis, ...]

    layout = layout.upper()
    if layout == "NCHW":
        tensor = np.transpose(tensor, (0, 3, 1, 2))
    elif layout == "NHWC":
        pass
    else:
        raise ValueError("Unsupported layout '{}', expected NCHW or NHWC".format(layout))

    tensor = tensor.astype(dtype, copy=False)

    if normalize:
        if not copy:
            tensor = tensor.copy()
        tensor *= scale
        if mean is not None:
            tensor -= np.asarray(mean, dtype=tensor.dtype)
        if std is not None:
            tensor /= np.asarray(std, dtype=tensor.dtype)

    return tensor
