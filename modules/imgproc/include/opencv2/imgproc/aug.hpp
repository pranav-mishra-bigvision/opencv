// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_IMGPROC_AUG_HPP
#define OPENCV_IMGPROC_AUG_HPP

#include "opencv2/imgproc.hpp"

namespace cv {
namespace aug {

class CV_EXPORTS AugmentationReplay;

//! @addtogroup imgproc_aug
//! @{

/** @brief Function-style randomized horizontal flip.

If sampling resolves to true, the image is flipped with `flipCode` semantics, otherwise copied.

@param src Source image.
@param dst Destination image.
@param probability Probability in [0, 1] of applying the flip.
@param flipCode Same semantics as cv::flip.
@param seed Optional deterministic seed. Use 0 for implementation-defined non-deterministic seeding.
*/
CV_EXPORTS_W void randomFlip(
        InputArray src,
        OutputArray dst,
        double probability = 0.5,
        int flipCode = 1,
        uint64 seed = 0);

CV_EXPORTS void randomFlip(
        InputArray src,
        OutputArray dst,
        double probability,
        int flipCode,
        uint64 seed,
        AugmentationReplay* replay);

/** @brief Function-style randomized affine transform.

Sampling policy is implementation-defined for this initial header and must be documented by concrete implementation.

@param src Source image.
@param dst Destination image.
@param maxRotateDeg Inclusive upper bound for absolute rotation in degrees.
@param maxTranslateX Inclusive upper bound for absolute horizontal translation in pixels.
@param maxTranslateY Inclusive upper bound for absolute vertical translation in pixels.
@param maxScaleDelta Inclusive upper bound for additive isotropic scale delta.
@param interpolation Interpolation mode (InterpolationFlags).
@param borderMode Border mode (BorderTypes).
@param borderValue Border value used with BORDER_CONSTANT.
@param seed Optional deterministic seed. Use 0 for implementation-defined non-deterministic seeding.
*/
CV_EXPORTS_W void randomAffine(
        InputArray src,
        OutputArray dst,
        double maxRotateDeg,
        double maxTranslateX,
        double maxTranslateY,
        double maxScaleDelta,
        int interpolation = INTER_LINEAR,
        int borderMode = BORDER_REFLECT_101,
        const Scalar& borderValue = Scalar(),
        uint64 seed = 0);

CV_EXPORTS void randomAffine(
        InputArray src,
        OutputArray dst,
        double maxRotateDeg,
        double maxTranslateX,
        double maxTranslateY,
        double maxScaleDelta,
        int interpolation,
        int borderMode,
        const Scalar& borderValue,
        uint64 seed,
        AugmentationReplay* replay);

CV_EXPORTS void randomAffine(
        InputArray src,
        OutputArray dst,
        double maxRotateDeg,
        double maxTranslateX,
        double maxTranslateY,
        double maxScaleDelta,
        double maxShearX,
        double maxShearY,
        int interpolation,
        int borderMode,
        const Scalar& borderValue,
        uint64 seed,
        AugmentationReplay* replay);

CV_EXPORTS void randomPerspectiveRemap(
        InputArray src,
        OutputArray dst,
        double maxJitterX,
        double maxJitterY,
        int interpolation,
        int borderMode,
        const Scalar& borderValue,
        uint64 seed,
        AugmentationReplay* replay);

/** @brief Function-style randomized brightness/contrast adjustment.

Depth/channel behavior:
- Supported depths: `CV_8U`, `CV_16U`, `CV_32F`.
- Supported layouts: grayscale (1 channel), BGR (3 channels), BGRA (4 channels).
- Processing is performed in normalized [0,1] domain and converted back to the source depth.
  - `CV_8U`: value range maps to [0,255].
  - `CV_16U`: value range maps to [0,65535].
  - `CV_32F`: values are treated directly as normalized [0,1].

@param src Source image.
@param dst Destination image.
@param maxBrightnessDelta Max absolute additive brightness delta in normalized units.
@param maxContrastDelta Max absolute contrast delta. Contrast multiplier is sampled from
       `[1-maxContrastDelta, 1+maxContrastDelta]` and clamped to non-negative.
@param seed Optional deterministic seed. Use 0 for implementation-defined non-deterministic seeding.
*/
CV_EXPORTS_W void randomBrightnessContrast(
        InputArray src,
        OutputArray dst,
        double maxBrightnessDelta,
        double maxContrastDelta,
        uint64 seed = 0);

CV_EXPORTS void randomBrightnessContrast(
        InputArray src,
        OutputArray dst,
        double maxBrightnessDelta,
        double maxContrastDelta,
        uint64 seed,
        AugmentationReplay* replay);

/** @brief Function-style randomized gamma correction.

Depth/channel behavior follows @ref randomBrightnessContrast.

@param src Source image.
@param dst Destination image.
@param maxGammaDelta Max absolute gamma delta. Gamma is sampled from
       `[1-maxGammaDelta, 1+maxGammaDelta]` and clamped to a small positive value.
@param seed Optional deterministic seed. Use 0 for implementation-defined non-deterministic seeding.
*/
CV_EXPORTS_W void randomGamma(
        InputArray src,
        OutputArray dst,
        double maxGammaDelta,
        uint64 seed = 0);

CV_EXPORTS void randomGamma(
        InputArray src,
        OutputArray dst,
        double maxGammaDelta,
        uint64 seed,
        AugmentationReplay* replay);

/** @brief Function-style randomized per-channel jitter (scale and bias).

Depth/channel behavior follows @ref randomBrightnessContrast.

@param src Source image.
@param dst Destination image.
@param maxScaleDelta Max absolute per-channel multiplicative delta around 1.
@param maxBiasDelta Max absolute per-channel additive delta in normalized units.
@param seed Optional deterministic seed. Use 0 for implementation-defined non-deterministic seeding.
*/
CV_EXPORTS_W void randomColorJitter(
        InputArray src,
        OutputArray dst,
        double maxScaleDelta,
        double maxBiasDelta,
        uint64 seed = 0);

CV_EXPORTS void randomColorJitter(
        InputArray src,
        OutputArray dst,
        double maxScaleDelta,
        double maxBiasDelta,
        uint64 seed,
        AugmentationReplay* replay);

/** @brief Function-style randomized additive Gaussian noise.

Depth/channel behavior follows @ref randomBrightnessContrast.

@param src Source image.
@param dst Destination image.
@param maxStdDev Max noise standard deviation in normalized [0,1] units.
@param seed Optional deterministic seed. Use 0 for implementation-defined non-deterministic seeding.
*/
CV_EXPORTS_W void randomGaussianNoise(
        InputArray src,
        OutputArray dst,
        double maxStdDev,
        uint64 seed = 0);

CV_EXPORTS void randomGaussianNoise(
        InputArray src,
        OutputArray dst,
        double maxStdDev,
        uint64 seed,
        AugmentationReplay* replay);

/** @brief Function-style randomized blur.

Depth/channel behavior follows @ref randomBrightnessContrast.

@param src Source image.
@param dst Destination image.
@param maxKernelRadius Max blur radius. Actual odd kernel size is sampled from `1..(2*maxKernelRadius+1)`.
@param seed Optional deterministic seed. Use 0 for implementation-defined non-deterministic seeding.
*/
CV_EXPORTS_W void randomBlur(
        InputArray src,
        OutputArray dst,
        int maxKernelRadius,
        uint64 seed = 0);

CV_EXPORTS void randomBlur(
        InputArray src,
        OutputArray dst,
        int maxKernelRadius,
        uint64 seed,
        AugmentationReplay* replay);

/** @brief Function-style randomized channel shuffle.

Supports BGR (3 channels) and BGRA (4 channels). For BGRA, only BGR channels are shuffled;
alpha is preserved. Grayscale inputs are copied unchanged.

@param src Source image.
@param dst Destination image.
@param probability Probability in [0,1] of applying the channel shuffle.
@param seed Optional deterministic seed. Use 0 for implementation-defined non-deterministic seeding.
*/
CV_EXPORTS_W void randomChannelShuffle(
        InputArray src,
        OutputArray dst,
        double probability = 0.5,
        uint64 seed = 0);

CV_EXPORTS void randomChannelShuffle(
        InputArray src,
        OutputArray dst,
        double probability,
        uint64 seed,
        AugmentationReplay* replay);

/** @brief Function-style randomized perspective transform.

Builds a quadrilateral by jittering each source-image corner and warps it to the original image
extent.

@param src Source image.
@param dst Destination image.
@param maxJitterX Inclusive upper bound for absolute horizontal corner jitter in pixels.
@param maxJitterY Inclusive upper bound for absolute vertical corner jitter in pixels.
@param interpolation Interpolation mode (InterpolationFlags).
@param borderMode Border mode (BorderTypes).
@param borderValue Border value used with BORDER_CONSTANT.
@param seed Optional deterministic seed. Use 0 for implementation-defined non-deterministic seeding.
*/
CV_EXPORTS_W void randomPerspective(
        InputArray src,
        OutputArray dst,
        double maxJitterX,
        double maxJitterY,
        int interpolation = INTER_LINEAR,
        int borderMode = BORDER_REFLECT_101,
        const Scalar& borderValue = Scalar(),
        uint64 seed = 0);

CV_EXPORTS void randomPerspective(
        InputArray src,
        OutputArray dst,
        double maxJitterX,
        double maxJitterY,
        int interpolation,
        int borderMode,
        const Scalar& borderValue,
        uint64 seed,
        AugmentationReplay* replay);

/** @brief Function-style randomized crop followed by resize.

The crop size is sampled as a fraction of source size and then resized to @p dsize.

@param src Source image.
@param dst Destination image.
@param minScale Minimal sampled crop scale relative to source side lengths.
@param maxScale Maximal sampled crop scale relative to source side lengths.
@param dsize Resize destination size.
@param interpolation Interpolation mode used by cv::resize.
@param seed Optional deterministic seed. Use 0 for implementation-defined non-deterministic seeding.
*/
CV_EXPORTS_W void randomCrop(
        InputArray src,
        OutputArray dst,
        double minScale,
        double maxScale,
        Size dsize,
        int interpolation = INTER_LINEAR,
        uint64 seed = 0);

CV_EXPORTS void randomCrop(
        InputArray src,
        OutputArray dst,
        double minScale,
        double maxScale,
        Size dsize,
        int interpolation,
        uint64 seed,
        AugmentationReplay* replay);

/** @brief Optional replay record used by randomized augmentations.

The record captures per-transform samples and can be reused to replay the exact same sampling sequence.
Serialization is done through cv::FileStorage as:
`{ format_version: 1, global_seed: <uint64>, transforms: [ { name: <string>, samples: [<double> ...] }, ... ] }`.

Determinism guarantees:
- With non-zero seed and no replay object, generated samples are deterministic for the same transform order.
- With a populated replay object, sampling is deterministic and independent from seed value.
- Pixel-level image results may still differ across platforms/builds due to floating-point/interpolation backends.
*/
class CV_EXPORTS AugmentationReplay
{
public:
    AugmentationReplay();
    ~AugmentationReplay();

    void clear();
    bool empty() const;

#ifndef CV_DOXYGEN
    struct Impl;
    Ptr<Impl> impl;
#endif
};

/** @brief Class-style augmentation operation.

Implementations are expected to be immutable configuration objects.
*/
class CV_EXPORTS_W AugmentationExecutionContext
{
public:
    enum CV_EXPORTS_W CoordinateConvention
    {
        COORDINATES_PIXEL_CENTERS = 0,
        COORDINATES_PIXEL_CORNERS = 1
    };

    /** @brief Mask target policy.

Masks default to nearest-neighbor interpolation to preserve label integrity.
*/
    struct CV_EXPORTS_W MaskTargetInfo
    {
        int interpolation;

        MaskTargetInfo();
    };

    /** @brief Keypoint target policy.

By default keypoints use pixel-center coordinates, are clipped to the image extent,
and are marked invisible when transformed outside the image.
*/
    struct CV_EXPORTS_W KeypointTargetInfo
    {
        CoordinateConvention convention;
        bool clipToImage;
        bool markInvisibleWhenOutside;

        KeypointTargetInfo();
    };

    /** @brief One sampled geometric transform shared across image-associated targets.

When `sampled` is true, `matrix` stores a forward transform in image coordinates
(from source to destination) and should be reused for image/mask/keypoint targets.
*/
    struct CV_EXPORTS_W GeometricTransformSample
    {
        Matx33d matrix;
        bool sampled;

        GeometricTransformSample();
    };

    struct CV_EXPORTS_W TargetInfo
    {
        Size size;
        int type;
        int channels;
        MaskTargetInfo mask;
        KeypointTargetInfo keypoints;

        TargetInfo();
    };

    AugmentationExecutionContext(RNG& rng, AugmentationReplay* replay = NULL);

    RNG& rng;
    AugmentationReplay* replay;
    TargetInfo target;
    GeometricTransformSample geometric;
};

class CV_EXPORTS_W AugmentationOp
{
public:
    virtual ~AugmentationOp();

    /** @brief Apply the operation with explicit execution context.

    @param src Source image.
    @param dst Destination image.
    @param ctx Execution context carrying RNG, replay data, and target metadata.
    */
    virtual void apply(InputArray src, OutputArray dst, AugmentationExecutionContext& ctx) const;

    /** @brief Apply the operation with explicit RNG state.

    @param src Source image.
    @param dst Destination image.
    @param rng RNG state consumed by the operation.
    */
    virtual void apply(InputArray src, OutputArray dst, RNG& rng) const = 0;
};

/** @brief Class-style sequential augmentation pipeline.

Operations are executed in insertion order; RNG is consumed in the same order.
*/
class CV_EXPORTS_W AugmentationPipeline
{
public:
    AugmentationPipeline();
    ~AugmentationPipeline();

    /** @brief Append an operation to the pipeline.

    @param op Shared operation instance.
    */
    CV_WRAP AugmentationPipeline& add(const Ptr<AugmentationOp>& op);

    /** @brief Append a named operation node with per-node probability.

    The probability is sampled exactly once per node during pipeline execution.

    @param name Stable node name used for replay record matching.
    @param op Shared operation instance.
    @param probability Probability in [0,1] of executing this node.
    */
    CV_WRAP AugmentationPipeline& add(const String& name, const Ptr<AugmentationOp>& op, double probability = 1.0);

    /** @brief Apply all operations using explicit RNG state.

    @param src Source image.
    @param dst Destination image.
    @param rng RNG state consumed by pipeline operations.
    */
    CV_WRAP void apply(InputArray src, OutputArray dst, RNG& rng) const;

    CV_EXPORTS void apply(InputArray src, OutputArray dst, RNG& rng, AugmentationReplay* replay) const;

    CV_EXPORTS void apply(InputArray src, OutputArray dst, AugmentationExecutionContext& ctx) const;

    /** @brief Apply all operations using optional deterministic seed.

    @param src Source image.
    @param dst Destination image.
    @param seed Optional deterministic seed. Use 0 for implementation-defined non-deterministic seeding.
    */
    CV_WRAP void apply(InputArray src, OutputArray dst, uint64 seed = 0) const;

    CV_EXPORTS void apply(InputArray src, OutputArray dst, uint64 seed, AugmentationReplay* replay) const;

#ifndef CV_DOXYGEN
    struct Impl;
    Ptr<Impl> impl;
#endif
};

//! @} imgproc_aug

} // namespace aug

CV_EXPORTS void write(FileStorage& fs, const String&, const aug::AugmentationReplay& x);
CV_EXPORTS void read(const FileNode& node, aug::AugmentationReplay& x, const aug::AugmentationReplay& default_value = aug::AugmentationReplay());

} // namespace cv

#endif // OPENCV_IMGPROC_AUG_HPP
