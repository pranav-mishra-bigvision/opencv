// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_IMGPROC_AUG_HPP
#define OPENCV_IMGPROC_AUG_HPP

#include "opencv2/imgproc.hpp"

namespace cv {
namespace aug {

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

    /** @brief Apply all operations using optional deterministic seed.

    @param src Source image.
    @param dst Destination image.
    @param seed Optional deterministic seed. Use 0 for implementation-defined non-deterministic seeding.
    */
    CV_WRAP void apply(InputArray src, OutputArray dst, uint64 seed = 0) const;

#ifndef CV_DOXYGEN
    struct Impl;
    Ptr<Impl> impl;
#endif
};

//! @} imgproc_aug

} // namespace aug
} // namespace cv

#endif // OPENCV_IMGPROC_AUG_HPP
