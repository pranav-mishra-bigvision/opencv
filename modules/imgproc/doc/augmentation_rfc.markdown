Augmentation API RFC (imgproc)
==============================

Status: Draft (API lock for initial implementation)

## 1. Scope and module placement decision

**Decision:** image augmentation stays in `imgproc` for the first public release, under namespace `cv::aug`.

### Rationale

- Existing transforms (flip/warp/resize/filter/color conversion) already live in `imgproc`; augmentation composes these primitives.
- Keeping augmentation in `imgproc` avoids additional module dependency and packaging surface during early stabilization.
- A dedicated top-level `aug` module remains possible later if:
  - runtime backends diverge from `imgproc`, or
  - augmentation-specific graph/runtime infrastructure is introduced.

### Consequence

- Public headers are placed under `opencv2/imgproc/`.
- Primary umbrella inclusion is exposed through `opencv2/imgproc.hpp`.

## 2. Namespace and API shape

### Namespace lock

All new public APIs are under:

- `cv::aug`.

### API split lock

The API is intentionally split into two forms:

1. **Function-style API** for one-shot stateless calls:
   - simple wrappers around a single randomized transform or fixed-policy transform.
   - optimized for scripting/binding ergonomics.

2. **Class-style API** for reusable policies/pipelines:
   - object instances carry immutable configuration.
   - execution takes explicit RNG state for deterministic replay.
   - composable via `AugmentationPipeline`.

This split is considered stable for the initial rollout.

## 3. Datatypes and tensor/image constraints

### Accepted image containers

- `InputArray` / `OutputArray` are the canonical public ABI types.
- Initial implementation targets `Mat` and `UMat` compatibility as supported by underlying imgproc kernels.

### Depth and channel constraints

- For the initial augmentation API, supported source depths are `CV_8U`, `CV_16U`, and `CV_32F`.
- `CV_8S`, `CV_16S`, `CV_32S`, and `CV_64F` are not supported and must fail with `Error::StsUnsupportedFormat`.
- Supported channel layouts are grayscale (1 channel), BGR (3 channels), and BGRA (4 channels).
- For BGRA inputs, intensity/color operations modify BGR channels and preserve alpha.

### Output typing rules

- By default, output type == input type.
- Any optional type override must follow existing imgproc conversion rules and use explicit parameterization.

## 4. Border and interpolation behavior

### Interpolation conventions

- Geometric augmentations accept interpolation as `int` with values from `InterpolationFlags`.
- Default interpolation for affine/perspective-style warps: `INTER_LINEAR`.
- Nearest-neighbor is allowed for masks/label maps and should be explicitly requested.

### Border conventions

- Border mode is an `int` using existing `BorderTypes`.
- Default border mode for geometric transforms: `BORDER_REFLECT_101`.
- `BORDER_CONSTANT` must require explicit `Scalar borderValue` handling.
- When behavior is undefined for a border/interpolation pair in existing imgproc kernels, APIs must fail fast with a descriptive `cv::Exception`.

## 5. RNG and determinism conventions

- Public APIs provide deterministic behavior when a seed is provided.
- Function-style APIs accept `uint64 seed` (default `0` means implementation-defined non-deterministic seed source).
- Class-style APIs execute with explicit `cv::RNG&` for deterministic composition and replay.
- Sampling distributions and parameter bounds must be documented per operation.
- For multi-op pipelines, RNG is advanced in declaration order of operations.

## 6. Error handling rules

- Invalid arguments must throw `cv::Exception` using standard OpenCV error codes:
  - `Error::StsBadArg` for invalid parameter ranges/types,
  - `Error::StsUnsupportedFormat` for unsupported depth/channel/layout,
  - `Error::StsNotImplemented` for declared-but-not-implemented backend paths.
- APIs must not silently clamp semantically invalid configuration.
- In-place operation is only permitted when explicitly documented per function/class; otherwise source and destination aliasing should throw `Error::StsBadArg`.

## 7. ABI and compatibility notes

- Public entry points are introduced via free functions and abstract interfaces in headers under `opencv2/imgproc/`.
- Virtual interfaces must use pImpl or pure-virtual ABI-stable facades; avoid exposing STL containers in data members.
- Do not remove or reorder existing virtual methods once released.
- New parameters must be appended (with defaults where possible) to preserve source compatibility.
- Behavioral changes that affect deterministic RNG streams are considered compatibility-significant and require release-note callouts.

## 8. Initial header plan

- Add `opencv2/imgproc/aug.hpp` containing:
  - function-style declarations,
  - class-style interfaces (`AugmentationOp`, `AugmentationPipeline`),
  - baseline execution and RNG contract docs.
- Include this header from `opencv2/imgproc.hpp` similarly to other imgproc extension headers.

## 9. Deferred topics (not locked by this RFC)

- Graph-level augmentation DSL.
- Batched tensor-native API beyond `InputArray`/`OutputArray`.
- Backend-specific acceleration policy (OpenCL, CUDA, oneDNN, etc.).
- Serialization format for augmentation pipelines.

## 10. Intensity/color operation behavior

- Normalized processing domain is `[0,1]`.
- `CV_8U`: values are normalized by `1/255` and converted back with saturation.
- `CV_16U`: values are normalized by `1/65535` and converted back with saturation.
- `CV_32F`: values are interpreted as normalized already; outputs are clamped to `[0,1]`.
- `randomChannelShuffle` applies only to BGR/BGRA; grayscale is a no-op copy. For BGRA, alpha is preserved.

## 11. Multi-target geometric contracts

- Geometric operations sample a single transform matrix per operation invocation.
- That sampled matrix is the authoritative transform and must be reused consistently across all related targets (image, masks, and keypoints) in the same execution step.
- Coordinate convention for keypoints is **pixel-center coordinates** by default.
- Keypoint policy defaults:
  - clip transformed points to image bounds,
  - mark keypoints invisible when transformed outside image bounds.
- Mask policy defaults to nearest-neighbor interpolation to preserve categorical labels.
