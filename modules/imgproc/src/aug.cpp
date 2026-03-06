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
                  uint64 seed)
{
    randomAffine(src, dst, maxRotateDeg, maxTranslateX, maxTranslateY, maxScaleDelta,
                 interpolation, borderMode, borderValue, seed, NULL);
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

    Point2f center(static_cast<float>(in.cols * 0.5), static_cast<float>(in.rows * 0.5));
    Mat M = getRotationMatrix2D(center, angleDeg, scale);
    M.at<double>(0, 2) += tx;
    M.at<double>(1, 2) += ty;

    warpAffine(in, dst, M, in.size(), interpolation, borderMode, borderValue);
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
