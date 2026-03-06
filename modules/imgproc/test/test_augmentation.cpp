// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/imgproc/aug.hpp"

namespace opencv_test {

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

} // namespace opencv_test
