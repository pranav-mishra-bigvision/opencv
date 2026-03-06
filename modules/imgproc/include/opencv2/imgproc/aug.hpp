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
class CV_EXPORTS_W AugmentationOp
{
public:
    virtual ~AugmentationOp();

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

    /** @brief Apply all operations using explicit RNG state.

    @param src Source image.
    @param dst Destination image.
    @param rng RNG state consumed by pipeline operations.
    */
    CV_WRAP void apply(InputArray src, OutputArray dst, RNG& rng) const;

    CV_EXPORTS void apply(InputArray src, OutputArray dst, RNG& rng, AugmentationReplay* replay) const;

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
