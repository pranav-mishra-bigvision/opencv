// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/imgproc/aug.hpp"

namespace opencv_test {

class AddValueOp : public cv::aug::AugmentationOp
{
public:
    explicit AddValueOp(int value) : value_(value) {}

    void apply(cv::InputArray src, cv::OutputArray dst, cv::RNG&) const CV_OVERRIDE
    {
        cv::Mat in = src.getMat();
        cv::add(in, cv::Scalar::all(value_), dst);
    }

private:
    int value_;
};

TEST(Imgproc_Augmentation, replay_randomFlip)
{
    cv::Mat src(16, 16, CV_8UC1);
    cv::randu(src, 0, 255);

    cv::aug::AugmentationReplay replay;
    cv::Mat out1, out2;

    cv::aug::randomFlip(src, out1, 0.5, 1, 12345, &replay);
    cv::aug::randomFlip(src, out2, 0.5, 1, 99999, &replay);

    EXPECT_EQ(0, cv::countNonZero(out1 != out2));
}

TEST(Imgproc_Augmentation, replay_roundtrip_FileStorage)
{
    cv::Mat src(32, 32, CV_8UC1);
    cv::randu(src, 0, 255);

    cv::aug::AugmentationReplay replayWrite;
    cv::Mat out1;
    cv::aug::randomAffine(src, out1, 15.0, 2.0, 3.0, 0.15, cv::INTER_LINEAR, cv::BORDER_REFLECT_101, cv::Scalar(), 2024, &replayWrite);

    cv::FileStorage fs("", cv::FileStorage::WRITE | cv::FileStorage::MEMORY);
    fs << "replay" << replayWrite;
    const cv::String serialized = fs.releaseAndGetString();

    cv::FileStorage fs2(serialized, cv::FileStorage::READ | cv::FileStorage::MEMORY);
    cv::aug::AugmentationReplay replayRead;
    fs2["replay"] >> replayRead;

    cv::Mat out2;
    cv::aug::randomAffine(src, out2, 15.0, 2.0, 3.0, 0.15, cv::INTER_LINEAR, cv::BORDER_REFLECT_101, cv::Scalar(), 1, &replayRead);

    EXPECT_EQ(0, cv::countNonZero(out1 != out2));
}

TEST(Imgproc_Augmentation, perspective_and_remap_match)
{
    cv::Mat src(64, 64, CV_8UC1);
    cv::randu(src, 0, 255);

    cv::Mat warpOut, remapOut;
    const uint64 seed = 4242;
    cv::aug::randomPerspective(src, warpOut, 4.0, 5.0,
                               cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(13), seed);
    cv::aug::randomPerspectiveRemap(src, remapOut, 4.0, 5.0,
                                    cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(13), seed, NULL);

    EXPECT_EQ(0, cv::countNonZero(warpOut != remapOut));
}

TEST(Imgproc_Augmentation, crop_replay)
{
    cv::Mat src(50, 80, CV_8UC3);
    cv::randu(src, 0, 255);

    cv::aug::AugmentationReplay replay;
    cv::Mat out1, out2;

    cv::aug::randomCrop(src, out1, 0.4, 0.9, cv::Size(32, 32), cv::INTER_LINEAR, 1001, &replay);
    cv::aug::randomCrop(src, out2, 0.4, 0.9, cv::Size(32, 32), cv::INTER_LINEAR, 5, &replay);

    EXPECT_EQ(32, out1.cols);
    EXPECT_EQ(32, out1.rows);
    EXPECT_EQ(0, cv::countNonZero(out1.reshape(1) != out2.reshape(1)));
}

TEST(Imgproc_Augmentation, affine_with_shear_replay)
{
    cv::Mat src(48, 48, CV_8UC1);
    cv::randu(src, 0, 255);

    cv::aug::AugmentationReplay replay;
    cv::Mat out1, out2;

    cv::aug::randomAffine(src, out1, 20.0, 5.0, 5.0, 0.2, 0.15, 0.1,
                          cv::INTER_LINEAR, cv::BORDER_REFLECT_101, cv::Scalar(), 333, &replay);
    cv::aug::randomAffine(src, out2, 20.0, 5.0, 5.0, 0.2, 0.15, 0.1,
                          cv::INTER_LINEAR, cv::BORDER_REFLECT_101, cv::Scalar(), 999, &replay);

    EXPECT_EQ(0, cv::countNonZero(out1 != out2));
}


TEST(Imgproc_Augmentation, color_ops_replay_and_layouts)
{
    cv::Mat src(24, 24, CV_16UC4);
    cv::randu(src, 0, 65535);

    cv::aug::AugmentationReplay replay;
    cv::Mat out1, out2;
    cv::aug::randomColorJitter(src, out1, 0.2, 0.05, 42, &replay);
    cv::aug::randomColorJitter(src, out2, 0.2, 0.05, 7, &replay);

    EXPECT_EQ(src.type(), out1.type());
    EXPECT_EQ(0, cv::countNonZero(out1.reshape(1) != out2.reshape(1)));
}

TEST(Imgproc_Augmentation, channel_shuffle_alpha_preserved)
{
    cv::Mat src(12, 14, CV_8UC4);
    cv::randu(src, 0, 255);

    std::vector<cv::Mat> sch;
    cv::split(src, sch);

    cv::Mat out;
    cv::aug::randomChannelShuffle(src, out, 1.0, 2025);

    std::vector<cv::Mat> och;
    cv::split(out, och);
    EXPECT_EQ(0, cv::countNonZero(sch[3] != och[3]));
}

TEST(Imgproc_Augmentation, invalid_args_and_channel_layout)
{
    cv::Mat badCh(8, 8, CV_8UC2);
    cv::Mat out;

    EXPECT_THROW(cv::aug::randomBrightnessContrast(badCh, out, 0.1, 0.1, 1), cv::Exception);

    cv::Mat src(8, 8, CV_8UC1);
    EXPECT_THROW(cv::aug::randomFlip(src, out, -0.1, 1, 1), cv::Exception);
    EXPECT_THROW(cv::aug::randomCrop(src, out, 0.0, 0.5, cv::Size(4, 4), cv::INTER_LINEAR, 1), cv::Exception);
    EXPECT_THROW(cv::aug::randomBlur(src, out, -1, 1), cv::Exception);
}

TEST(Imgproc_Augmentation, pipeline_replay_probability_and_order)
{
    cv::Mat src(8, 8, CV_8UC1, cv::Scalar(10));

    cv::aug::AugmentationPipeline p;
    p.add("always_add_3", cv::makePtr<AddValueOp>(3), 1.0);
    p.add("never_add_100", cv::makePtr<AddValueOp>(100), 0.0);
    p.add("always_add_5", cv::makePtr<AddValueOp>(5), 1.0);

    cv::aug::AugmentationReplay replay;
    cv::Mat out1, out2;
    p.apply(src, out1, 1234, &replay);
    p.apply(src, out2, 9876, &replay);

    EXPECT_EQ(0, cv::countNonZero(out1 != out2));
    EXPECT_EQ(0, cv::countNonZero(out1 != cv::Scalar(18)));
}

TEST(Imgproc_Augmentation, pipeline_add_named_validation)
{
    cv::aug::AugmentationPipeline p;
    cv::Ptr<cv::aug::AugmentationOp> op = cv::makePtr<AddValueOp>(1);

    EXPECT_THROW(p.add("", op, 1.0), cv::Exception);
    EXPECT_THROW(p.add("bad_probability", op, -0.1), cv::Exception);
    EXPECT_THROW(p.add("bad_probability", op, 1.1), cv::Exception);
}

} // namespace opencv_test
