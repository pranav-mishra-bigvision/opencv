// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

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
    TransformSampler sampler(replay, "randomFlip", rng);

    if (sampler.bernoulli(probability))
        flip(src, dst, flipCode);
    else
        src.copyTo(dst);
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
    CV_Assert(!in.empty());

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
    CV_Assert(!in.empty());

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
    CV_Assert(!in.empty());
    CV_Assert(dsize.width > 0 && dsize.height > 0);

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
    CV_Assert(!in.empty());

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
