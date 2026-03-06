// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include <algorithm>
#include <limits>
#include <sstream>

namespace cv {
namespace aug {

struct AugmentationReplay::Impl
{
    struct TransformRecord
    {
        String name;
        std::vector<double> samples;
        size_t sample_read_pos;

        TransformRecord() : sample_read_pos(0) {}
    };

    uint64 global_seed;
    int format_version;
    std::vector<TransformRecord> transforms;
    size_t transform_read_pos;

    Impl() : global_seed(0), format_version(1), transform_read_pos(0) {}

    bool hasReplayData() const
    {
        return !transforms.empty();
    }

    void resetReadState()
    {
        transform_read_pos = 0;
        for (size_t i = 0; i < transforms.size(); ++i)
            transforms[i].sample_read_pos = 0;
    }

    void clear()
    {
        global_seed = 0;
        format_version = 1;
        transforms.clear();
        transform_read_pos = 0;
    }
};

namespace {

struct WarpParams
{
    int interpolation;
    int borderMode;
    Scalar borderValue;

    WarpParams(int interpolation_, int borderMode_, const Scalar& borderValue_)
        : interpolation(interpolation_), borderMode(borderMode_), borderValue(borderValue_) {}
};

static void validateAugInput(const Mat& in, const char* fn, bool enforceDepth)
{
    if (in.empty())
        CV_Error(Error::StsBadArg, String(fn) + ": src must be non-empty");

    const int cn = in.channels();
    if (!(cn == 1 || cn == 3 || cn == 4))
        CV_Error(Error::StsUnsupportedFormat, String(fn) + ": supported channel layouts are grayscale (1), BGR (3), BGRA (4)");

    if (enforceDepth)
    {
        const int d = in.depth();
        if (!(d == CV_8U || d == CV_16U || d == CV_32F))
            CV_Error(Error::StsUnsupportedFormat, String(fn) + ": supported depths are CV_8U, CV_16U, CV_32F");
    }
}

static double getDepthScale(int depth)
{
    if (depth == CV_8U) return 1.0 / 255.0;
    if (depth == CV_16U) return 1.0 / 65535.0;
    return 1.0;
}

static void clampUnit(Mat& m)
{
    cv::max(m, 0.0, m);
    cv::min(m, 1.0, m);
}

static Mat toNormalizedFloat(const Mat& in)
{
    Mat f;
    in.convertTo(f, CV_32F, getDepthScale(in.depth()));
    return f;
}

static void fromNormalizedFloat(const Mat& f, int depth, OutputArray dst)
{
    Mat clamped = f.clone();
    clampUnit(clamped);
    if (depth == CV_8U)
        clamped.convertTo(dst, CV_8U, 255.0);
    else if (depth == CV_16U)
        clamped.convertTo(dst, CV_16U, 65535.0);
    else
        clamped.convertTo(dst, CV_32F);
}

static uint64 makeGlobalSeed(uint64 seed)
{
    if (seed != 0)
        return seed;
    const uint64 t = static_cast<uint64>(getTickCount());
    const uint64 f = static_cast<uint64>(getCPUTickCount());
    return t ^ (f + 0x9e3779b97f4a7c15ULL);
}

class TransformSampler
{
public:
    TransformSampler(AugmentationReplay* replay, const String& name, RNG& rng)
        : replay_(replay), rng_(rng), replayRecord_(false)
    {
        if (!replay_)
            return;

        if (!replay_->impl)
            replay_->impl = makePtr<AugmentationReplay::Impl>();

        AugmentationReplay::Impl& impl = *replay_->impl;
        if (impl.transform_read_pos < impl.transforms.size())
        {
            AugmentationReplay::Impl::TransformRecord& rec = impl.transforms[impl.transform_read_pos];
            CV_Assert(rec.name == name);
            record_ = &rec;
            impl.transform_read_pos++;
            replayRecord_ = true;
        }
        else
        {
            AugmentationReplay::Impl::TransformRecord rec;
            rec.name = name;
            impl.transforms.push_back(rec);
            record_ = &impl.transforms.back();
        }
    }

    double sampleUnit()
    {
        if (record_ && replayRecord_)
        {
            CV_Assert(record_->sample_read_pos < record_->samples.size());
            return record_->samples[record_->sample_read_pos++];
        }

        const double s = rng_.uniform(0.0, 1.0);
        if (record_)
            record_->samples.push_back(s);
        return s;
    }

    bool bernoulli(double probability)
    {
        probability = std::max(0.0, std::min(1.0, probability));
        return sampleUnit() < probability;
    }

    double symmetric(double maxAbs)
    {
        return (sampleUnit() * 2.0 - 1.0) * maxAbs;
    }

private:
    AugmentationReplay* replay_;
    RNG& rng_;
    AugmentationReplay::Impl::TransformRecord* record_ = NULL;
    bool replayRecord_;
};

static void initReplayForWrite(AugmentationReplay* replay, uint64 globalSeed)
{
    if (!replay)
        return;

    if (!replay->impl)
        replay->impl = makePtr<AugmentationReplay::Impl>();

    if (!replay->impl->hasReplayData())
    {
        replay->impl->global_seed = globalSeed;
        replay->impl->format_version = 1;
    }
    replay->impl->resetReadState();
}

static Matx33d makeTranslation(double tx, double ty)
{
    return Matx33d(1.0, 0.0, tx,
                   0.0, 1.0, ty,
                   0.0, 0.0, 1.0);
}

static Matx33d makeScale(double sx, double sy)
{
    return Matx33d(sx, 0.0, 0.0,
                   0.0, sy, 0.0,
                   0.0, 0.0, 1.0);
}

static Matx33d makeRotation(double angleRad)
{
    const double c = std::cos(angleRad);
    const double s = std::sin(angleRad);
    return Matx33d(c, -s, 0.0,
                   s,  c, 0.0,
                   0.0, 0.0, 1.0);
}

static Matx33d makeShear(double shx, double shy)
{
    return Matx33d(1.0, shx, 0.0,
                   shy, 1.0, 0.0,
                   0.0, 0.0, 1.0);
}

static Matx33d makeCenteredAffine(const Mat& in, double angleDeg, double tx, double ty,
                                  double sx, double sy, double shx, double shy)
{
    const Point2d center(in.cols * 0.5, in.rows * 0.5);
    const double angleRad = angleDeg * CV_PI / 180.0;
    return makeTranslation(center.x + tx, center.y + ty)
         * makeRotation(angleRad)
         * makeShear(shx, shy)
         * makeScale(sx, sy)
         * makeTranslation(-center.x, -center.y);
}

static Mat toMatAffine(const Matx33d& H)
{
    Mat M(2, 3, CV_64F);
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 3; ++x)
            M.at<double>(y, x) = H(y, x);
    return M;
}

static void applyAffineWarp(const Mat& in, OutputArray dst, const Matx33d& H, const WarpParams& wp)
{
    warpAffine(in, dst, toMatAffine(H), in.size(), wp.interpolation, wp.borderMode, wp.borderValue);
}

static void applyPerspectiveWarp(const Mat& in, OutputArray dst, const Matx33d& H, const WarpParams& wp)
{
    warpPerspective(in, dst, Mat(H), in.size(), wp.interpolation, wp.borderMode, wp.borderValue);
}

static void samplePerspectiveMaps(const Matx33d& H, const Size& dsize, Mat& mapX, Mat& mapY)
{
    mapX.create(dsize, CV_32FC1);
    mapY.create(dsize, CV_32FC1);

    for (int y = 0; y < dsize.height; ++y)
    {
        float* mx = mapX.ptr<float>(y);
        float* my = mapY.ptr<float>(y);
        for (int x = 0; x < dsize.width; ++x)
        {
            const double w = H(2, 0) * x + H(2, 1) * y + H(2, 2);
            const double iw = (std::abs(w) > std::numeric_limits<double>::epsilon()) ? 1.0 / w : 0.0;
            mx[x] = static_cast<float>((H(0, 0) * x + H(0, 1) * y + H(0, 2)) * iw);
            my[x] = static_cast<float>((H(1, 0) * x + H(1, 1) * y + H(1, 2)) * iw);
        }
    }
}

static void applyPerspectiveRemap(const Mat& in, OutputArray dst, const Matx33d& H, const WarpParams& wp)
{
    Mat mapX, mapY;
    samplePerspectiveMaps(H, in.size(), mapX, mapY);
    remap(in, dst, mapX, mapY, wp.interpolation, wp.borderMode, wp.borderValue);
}

} // namespace

AugmentationReplay::AugmentationReplay() : impl(makePtr<Impl>()) {}
AugmentationReplay::~AugmentationReplay() {}

void AugmentationReplay::clear()
{
    if (!impl)
        impl = makePtr<Impl>();
    impl->clear();
}

bool AugmentationReplay::empty() const
{
    return !impl || impl->transforms.empty();
}

AugmentationOp::~AugmentationOp() {}

struct AugmentationPipeline::Impl
{
    std::vector<Ptr<AugmentationOp> > ops;
};

AugmentationPipeline::AugmentationPipeline() : impl(makePtr<Impl>()) {}
AugmentationPipeline::~AugmentationPipeline() {}

AugmentationPipeline& AugmentationPipeline::add(const Ptr<AugmentationOp>& op)
{
    CV_Assert(!op.empty());
    impl->ops.push_back(op);
    return *this;
}

void AugmentationPipeline::apply(InputArray src, OutputArray dst, RNG& rng) const
{
    apply(src, dst, rng, NULL);
}

void AugmentationPipeline::apply(InputArray src, OutputArray dst, RNG& rng, AugmentationReplay* replay) const
{
    Mat in = src.getMat();
    Mat current = in;
    Mat tmp;

    initReplayForWrite(replay, 0);

    for (size_t i = 0; i < impl->ops.size(); ++i)
    {
        std::ostringstream ss;
        ss << "pipeline_op_" << i;
        TransformSampler sampler(replay, String(ss.str()), rng);

        const double s = sampler.sampleUnit();
        const uint64 childSeed = static_cast<uint64>(s * static_cast<double>(std::numeric_limits<uint64>::max()));
        RNG childRng(childSeed);

        impl->ops[i]->apply(current, tmp, childRng);
        current = tmp;
    }

    current.copyTo(dst);
}

void AugmentationPipeline::apply(InputArray src, OutputArray dst, uint64 seed) const
{
    apply(src, dst, seed, NULL);
}

void AugmentationPipeline::apply(InputArray src, OutputArray dst, uint64 seed, AugmentationReplay* replay) const
{
    const uint64 globalSeed = makeGlobalSeed(seed);
    RNG rng(globalSeed);
    initReplayForWrite(replay, globalSeed);
    apply(src, dst, rng, replay);
}

void randomFlip(InputArray src, OutputArray dst, double probability, int flipCode, uint64 seed)
{
    randomFlip(src, dst, probability, flipCode, seed, NULL);
}

void randomFlip(InputArray src, OutputArray dst, double probability, int flipCode, uint64 seed, AugmentationReplay* replay)
{
    const uint64 globalSeed = makeGlobalSeed(seed);
    RNG rng(globalSeed);
    initReplayForWrite(replay, globalSeed);
    if (probability < 0.0 || probability > 1.0)
        CV_Error(Error::StsBadArg, "randomFlip: probability must be in [0,1]");
    if (!(flipCode == -1 || flipCode == 0 || flipCode == 1))
        CV_Error(Error::StsBadArg, "randomFlip: flipCode must be -1, 0, or 1");

    TransformSampler sampler(replay, "randomFlip", rng);

    Mat in = src.getMat();
    validateAugInput(in, "randomFlip", true);

    if (sampler.bernoulli(probability))
        flip(in, dst, flipCode);
    else
        in.copyTo(dst);
}

void randomAffine(InputArray src, OutputArray dst,
                  double maxRotateDeg,
                  double maxTranslateX,
                  double maxTranslateY,
                  double maxScaleDelta,
                  int interpolation,
                  int borderMode,
                  const Scalar& borderValue,
                  uint64 seed,
                  AugmentationReplay* replay)
{
    randomAffine(src, dst, maxRotateDeg, maxTranslateX, maxTranslateY, maxScaleDelta,
                 0.0, 0.0, interpolation, borderMode, borderValue, seed, replay);
}

void randomAffine(InputArray src, OutputArray dst,
                  double maxRotateDeg,
                  double maxTranslateX,
                  double maxTranslateY,
                  double maxScaleDelta,
                  int interpolation,
                  int borderMode,
                  const Scalar& borderValue,
                  uint64 seed)
{
    randomAffine(src, dst, maxRotateDeg, maxTranslateX, maxTranslateY, maxScaleDelta,
                 0.0, 0.0, interpolation, borderMode, borderValue, seed, NULL);
}

void randomAffine(InputArray src, OutputArray dst,
                  double maxRotateDeg,
                  double maxTranslateX,
                  double maxTranslateY,
                  double maxScaleDelta,
                  double maxShearX,
                  double maxShearY,
                  int interpolation,
                  int borderMode,
                  const Scalar& borderValue,
                  uint64 seed)
{
    randomAffine(src, dst, maxRotateDeg, maxTranslateX, maxTranslateY, maxScaleDelta, maxShearX, maxShearY,
                 interpolation, borderMode, borderValue, seed, NULL);
}

void randomAffine(InputArray src, OutputArray dst,
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
                  AugmentationReplay* replay)
{
    Mat in = src.getMat();
    validateAugInput(in, "randomAffine", true);
    if (maxScaleDelta < 0.0)
        CV_Error(Error::StsBadArg, "randomAffine: maxScaleDelta must be non-negative");

    const uint64 globalSeed = makeGlobalSeed(seed);
    RNG rng(globalSeed);
    initReplayForWrite(replay, globalSeed);
    TransformSampler sampler(replay, "randomAffine", rng);

    const double angleDeg = sampler.symmetric(std::abs(maxRotateDeg));
    const double tx = sampler.symmetric(std::abs(maxTranslateX));
    const double ty = sampler.symmetric(std::abs(maxTranslateY));
    const double scale = 1.0 + sampler.symmetric(std::abs(maxScaleDelta));
    const double shx = sampler.symmetric(std::abs(maxShearX));
    const double shy = sampler.symmetric(std::abs(maxShearY));

    const Matx33d H = makeCenteredAffine(in, angleDeg, tx, ty, scale, scale, shx, shy);
    applyAffineWarp(in, dst, H, WarpParams(interpolation, borderMode, borderValue));
}

void randomPerspective(InputArray src, OutputArray dst,
                       double maxJitterX,
                       double maxJitterY,
                       int interpolation,
                       int borderMode,
                       const Scalar& borderValue,
                       uint64 seed)
{
    randomPerspective(src, dst, maxJitterX, maxJitterY, interpolation, borderMode, borderValue, seed, NULL);
}

void randomPerspective(InputArray src, OutputArray dst,
                       double maxJitterX,
                       double maxJitterY,
                       int interpolation,
                       int borderMode,
                       const Scalar& borderValue,
                       uint64 seed,
                       AugmentationReplay* replay)
{
    Mat in = src.getMat();
    validateAugInput(in, "randomPerspective", true);
    if (maxJitterX < 0.0 || maxJitterY < 0.0)
        CV_Error(Error::StsBadArg, "randomPerspective: max jitter must be non-negative");

    const uint64 globalSeed = makeGlobalSeed(seed);
    RNG rng(globalSeed);
    initReplayForWrite(replay, globalSeed);
    TransformSampler sampler(replay, "randomPerspective", rng);

    const double jx = std::abs(maxJitterX);
    const double jy = std::abs(maxJitterY);
    std::vector<Point2f> srcPts(4), dstPts(4);
    srcPts[0] = Point2f(0.f, 0.f);
    srcPts[1] = Point2f(static_cast<float>(in.cols - 1), 0.f);
    srcPts[2] = Point2f(static_cast<float>(in.cols - 1), static_cast<float>(in.rows - 1));
    srcPts[3] = Point2f(0.f, static_cast<float>(in.rows - 1));
    for (int i = 0; i < 4; ++i)
        dstPts[i] = Point2f(static_cast<float>(srcPts[i].x + sampler.symmetric(jx)),
                            static_cast<float>(srcPts[i].y + sampler.symmetric(jy)));

    const Matx33d H = Matx33d(getPerspectiveTransform(srcPts, dstPts));
    applyPerspectiveWarp(in, dst, H, WarpParams(interpolation, borderMode, borderValue));
}

void randomCrop(InputArray src, OutputArray dst,
                double minScale,
                double maxScale,
                Size dsize,
                int interpolation,
                uint64 seed)
{
    randomCrop(src, dst, minScale, maxScale, dsize, interpolation, seed, NULL);
}

void randomCrop(InputArray src, OutputArray dst,
                double minScale,
                double maxScale,
                Size dsize,
                int interpolation,
                uint64 seed,
                AugmentationReplay* replay)
{
    Mat in = src.getMat();
    validateAugInput(in, "randomCrop", true);
    if (dsize.width <= 0 || dsize.height <= 0)
        CV_Error(Error::StsBadArg, "randomCrop: dsize must be positive");
    if (minScale <= 0.0 || maxScale <= 0.0)
        CV_Error(Error::StsBadArg, "randomCrop: minScale and maxScale must be positive");

    const uint64 globalSeed = makeGlobalSeed(seed);
    RNG rng(globalSeed);
    initReplayForWrite(replay, globalSeed);
    TransformSampler sampler(replay, "randomCrop", rng);

    const double lo = std::min(minScale, maxScale);
    const double hi = std::max(minScale, maxScale);
    const double scaleFactor = std::max(1e-6, lo + sampler.sampleUnit() * (hi - lo));

    const int cw = std::max(1, std::min(in.cols, cvRound(in.cols * scaleFactor)));
    const int ch = std::max(1, std::min(in.rows, cvRound(in.rows * scaleFactor)));
    const int maxX = in.cols - cw;
    const int maxY = in.rows - ch;
    const int x = (maxX > 0) ? cvFloor(sampler.sampleUnit() * (maxX + 1)) : 0;
    const int y = (maxY > 0) ? cvFloor(sampler.sampleUnit() * (maxY + 1)) : 0;

    resize(in(Rect(x, y, cw, ch)), dst, dsize, 0.0, 0.0, interpolation);
}

void randomPerspectiveRemap(InputArray src, OutputArray dst,
                            double maxJitterX,
                            double maxJitterY,
                            int interpolation,
                            int borderMode,
                            const Scalar& borderValue,
                            uint64 seed,
                            AugmentationReplay* replay)
{
    Mat in = src.getMat();
    validateAugInput(in, "randomPerspectiveRemap", true);

    if (maxJitterX < 0.0 || maxJitterY < 0.0)
        CV_Error(Error::StsBadArg, "randomPerspectiveRemap: max jitter must be non-negative");

    const uint64 globalSeed = makeGlobalSeed(seed);
    RNG rng(globalSeed);
    initReplayForWrite(replay, globalSeed);
    TransformSampler sampler(replay, "randomPerspectiveRemap", rng);

    const double jx = std::abs(maxJitterX);
    const double jy = std::abs(maxJitterY);
    std::vector<Point2f> srcPts(4), dstPts(4);
    srcPts[0] = Point2f(0.f, 0.f);
    srcPts[1] = Point2f(static_cast<float>(in.cols - 1), 0.f);
    srcPts[2] = Point2f(static_cast<float>(in.cols - 1), static_cast<float>(in.rows - 1));
    srcPts[3] = Point2f(0.f, static_cast<float>(in.rows - 1));
    for (int i = 0; i < 4; ++i)
        dstPts[i] = Point2f(static_cast<float>(srcPts[i].x + sampler.symmetric(jx)),
                            static_cast<float>(srcPts[i].y + sampler.symmetric(jy)));

    const Matx33d H = Matx33d(getPerspectiveTransform(dstPts, srcPts));
    applyPerspectiveRemap(in, dst, H, WarpParams(interpolation, borderMode, borderValue));
}


void randomBrightnessContrast(InputArray src, OutputArray dst,
                              double maxBrightnessDelta,
                              double maxContrastDelta,
                              uint64 seed)
{
    randomBrightnessContrast(src, dst, maxBrightnessDelta, maxContrastDelta, seed, NULL);
}

void randomBrightnessContrast(InputArray src, OutputArray dst,
                              double maxBrightnessDelta,
                              double maxContrastDelta,
                              uint64 seed,
                              AugmentationReplay* replay)
{
    if (maxBrightnessDelta < 0.0 || maxContrastDelta < 0.0)
        CV_Error(Error::StsBadArg, "randomBrightnessContrast: max deltas must be non-negative");
    Mat in = src.getMat();
    validateAugInput(in, "randomBrightnessContrast", true);

    const uint64 globalSeed = makeGlobalSeed(seed);
    RNG rng(globalSeed);
    initReplayForWrite(replay, globalSeed);
    TransformSampler sampler(replay, "randomBrightnessContrast", rng);

    const double brightness = sampler.symmetric(maxBrightnessDelta);
    const double contrast = std::max(0.0, 1.0 + sampler.symmetric(maxContrastDelta));

    Mat f = toNormalizedFloat(in);
    std::vector<Mat> ch;
    split(f, ch);
    const int workCn = (in.channels() == 4) ? 3 : in.channels();
    for (int i = 0; i < workCn; ++i)
        ch[i] = (ch[i] - 0.5f) * contrast + (0.5f + brightness);
    merge(ch, f);
    fromNormalizedFloat(f, in.depth(), dst);
}

void randomGamma(InputArray src, OutputArray dst, double maxGammaDelta, uint64 seed)
{
    randomGamma(src, dst, maxGammaDelta, seed, NULL);
}

void randomGamma(InputArray src, OutputArray dst, double maxGammaDelta, uint64 seed, AugmentationReplay* replay)
{
    if (maxGammaDelta < 0.0)
        CV_Error(Error::StsBadArg, "randomGamma: maxGammaDelta must be non-negative");
    Mat in = src.getMat();
    validateAugInput(in, "randomGamma", true);

    const uint64 globalSeed = makeGlobalSeed(seed);
    RNG rng(globalSeed);
    initReplayForWrite(replay, globalSeed);
    TransformSampler sampler(replay, "randomGamma", rng);

    const float gamma = static_cast<float>(std::max(1e-6, 1.0 + sampler.symmetric(maxGammaDelta)));
    Mat f = toNormalizedFloat(in);
    std::vector<Mat> ch;
    split(f, ch);
    const int workCn = (in.channels() == 4) ? 3 : in.channels();
    for (int i = 0; i < workCn; ++i)
        cv::pow(ch[i], gamma, ch[i]);
    merge(ch, f);
    fromNormalizedFloat(f, in.depth(), dst);
}

void randomColorJitter(InputArray src, OutputArray dst,
                       double maxScaleDelta,
                       double maxBiasDelta,
                       uint64 seed)
{
    randomColorJitter(src, dst, maxScaleDelta, maxBiasDelta, seed, NULL);
}

void randomColorJitter(InputArray src, OutputArray dst,
                       double maxScaleDelta,
                       double maxBiasDelta,
                       uint64 seed,
                       AugmentationReplay* replay)
{
    if (maxScaleDelta < 0.0 || maxBiasDelta < 0.0)
        CV_Error(Error::StsBadArg, "randomColorJitter: max deltas must be non-negative");
    Mat in = src.getMat();
    validateAugInput(in, "randomColorJitter", true);

    const uint64 globalSeed = makeGlobalSeed(seed);
    RNG rng(globalSeed);
    initReplayForWrite(replay, globalSeed);
    TransformSampler sampler(replay, "randomColorJitter", rng);

    Mat f = toNormalizedFloat(in);
    std::vector<Mat> ch;
    split(f, ch);
    const int workCn = (in.channels() == 4) ? 3 : in.channels();
    for (int i = 0; i < workCn; ++i)
    {
        const float scale = static_cast<float>(std::max(0.0, 1.0 + sampler.symmetric(maxScaleDelta)));
        const float bias = static_cast<float>(sampler.symmetric(maxBiasDelta));
        ch[i] = ch[i] * scale + bias;
    }
    merge(ch, f);
    fromNormalizedFloat(f, in.depth(), dst);
}

void randomGaussianNoise(InputArray src, OutputArray dst, double maxStdDev, uint64 seed)
{
    randomGaussianNoise(src, dst, maxStdDev, seed, NULL);
}

void randomGaussianNoise(InputArray src, OutputArray dst, double maxStdDev, uint64 seed, AugmentationReplay* replay)
{
    if (maxStdDev < 0.0)
        CV_Error(Error::StsBadArg, "randomGaussianNoise: maxStdDev must be non-negative");
    Mat in = src.getMat();
    validateAugInput(in, "randomGaussianNoise", true);

    const uint64 globalSeed = makeGlobalSeed(seed);
    RNG rng(globalSeed);
    initReplayForWrite(replay, globalSeed);
    TransformSampler sampler(replay, "randomGaussianNoise", rng);

    const double stddev = sampler.sampleUnit() * maxStdDev;
    const uint64 noiseSeed = static_cast<uint64>(sampler.sampleUnit() * static_cast<double>(std::numeric_limits<uint64>::max()));
    RNG noiseRng(noiseSeed);

    Mat f = toNormalizedFloat(in);
    Mat n(f.size(), f.type());
    noiseRng.fill(n, RNG::NORMAL, Scalar::all(0), Scalar::all(stddev));
    if (in.channels() == 4)
    {
        std::vector<Mat> fc, nc;
        split(f, fc);
        split(n, nc);
        for (int i = 0; i < 3; ++i)
            fc[i] += nc[i];
        merge(fc, f);
    }
    else
    {
        f += n;
    }
    fromNormalizedFloat(f, in.depth(), dst);
}

void randomBlur(InputArray src, OutputArray dst, int maxKernelRadius, uint64 seed)
{
    randomBlur(src, dst, maxKernelRadius, seed, NULL);
}

void randomBlur(InputArray src, OutputArray dst, int maxKernelRadius, uint64 seed, AugmentationReplay* replay)
{
    if (maxKernelRadius < 0)
        CV_Error(Error::StsBadArg, "randomBlur: maxKernelRadius must be non-negative");
    Mat in = src.getMat();
    validateAugInput(in, "randomBlur", true);

    const uint64 globalSeed = makeGlobalSeed(seed);
    RNG rng(globalSeed);
    initReplayForWrite(replay, globalSeed);
    TransformSampler sampler(replay, "randomBlur", rng);

    const int radius = (maxKernelRadius > 0) ? cvFloor(sampler.sampleUnit() * (maxKernelRadius + 1)) : 0;
    const int ksize = radius * 2 + 1;
    if (ksize <= 1)
    {
        in.copyTo(dst);
        return;
    }
    GaussianBlur(in, dst, Size(ksize, ksize), 0.0, 0.0, BORDER_REFLECT_101);
}

void randomChannelShuffle(InputArray src, OutputArray dst, double probability, uint64 seed)
{
    randomChannelShuffle(src, dst, probability, seed, NULL);
}

void randomChannelShuffle(InputArray src, OutputArray dst, double probability, uint64 seed, AugmentationReplay* replay)
{
    if (probability < 0.0 || probability > 1.0)
        CV_Error(Error::StsBadArg, "randomChannelShuffle: probability must be in [0,1]");
    Mat in = src.getMat();
    validateAugInput(in, "randomChannelShuffle", true);

    const uint64 globalSeed = makeGlobalSeed(seed);
    RNG rng(globalSeed);
    initReplayForWrite(replay, globalSeed);
    TransformSampler sampler(replay, "randomChannelShuffle", rng);

    if (in.channels() == 1 || !sampler.bernoulli(probability))
    {
        in.copyTo(dst);
        return;
    }

    static const int perms[6][3] = {
        {0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0}
    };
    const int permId = std::min(5, cvFloor(sampler.sampleUnit() * 6.0));

    std::vector<Mat> ch;
    split(in, ch);
    std::vector<Mat> out(ch.size());
    for (size_t i = 0; i < ch.size(); ++i)
        out[i] = ch[i];
    out[0] = ch[perms[permId][0]];
    out[1] = ch[perms[permId][1]];
    out[2] = ch[perms[permId][2]];
    merge(out, dst);
}

} // namespace aug

void write(FileStorage& fs, const String&, const aug::AugmentationReplay& x)
{
    fs << "{";
    fs << "format_version" << 1;

    const bool hasImpl = !x.impl.empty();
    fs << "global_seed" << static_cast<int64>(hasImpl ? x.impl->global_seed : 0);

    fs << "transforms" << "[";
    if (hasImpl)
    {
        for (size_t i = 0; i < x.impl->transforms.size(); ++i)
        {
            const aug::AugmentationReplay::Impl::TransformRecord& rec = x.impl->transforms[i];
            fs << "{";
            fs << "name" << rec.name;
            fs << "samples" << "[";
            for (size_t j = 0; j < rec.samples.size(); ++j)
                fs << rec.samples[j];
            fs << "]";
            fs << "}";
        }
    }
    fs << "]";
    fs << "}";
}

void read(const FileNode& node, aug::AugmentationReplay& x, const aug::AugmentationReplay& default_value)
{
    if (node.empty())
    {
        x = default_value;
        return;
    }

    if (x.impl.empty())
        x.impl = makePtr<aug::AugmentationReplay::Impl>();
    x.impl->clear();

    x.impl->format_version = static_cast<int>((int)node["format_version"]);
    x.impl->global_seed = static_cast<uint64>((int64)node["global_seed"]);

    FileNode transforms = node["transforms"];
    if (transforms.type() == FileNode::SEQ)
    {
        for (FileNodeIterator it = transforms.begin(); it != transforms.end(); ++it)
        {
            aug::AugmentationReplay::Impl::TransformRecord rec;
            rec.name = static_cast<String>((*it)["name"]);

            FileNode samples = (*it)["samples"];
            if (samples.type() == FileNode::SEQ)
            {
                for (FileNodeIterator sit = samples.begin(); sit != samples.end(); ++sit)
                    rec.samples.push_back((double)*sit);
            }
            x.impl->transforms.push_back(rec);
        }
    }
    x.impl->resetReadState();
}

} // namespace cv
