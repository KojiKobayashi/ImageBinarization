#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>
#include "../src/NiblackBinarization.h"

using namespace ImageBinarization;

TEST(NiblackBinarization, ALLPixelsAre0Or255)
{
    const auto fileName = "images/Lenna.png";
    auto src = cv::imread(fileName, 0);
    cv::Mat dst;
    NiblackBinarization::Binarize(src, dst, 10);

    auto area = src.rows * src.cols;
    auto blackCount = area - cv::countNonZero(dst);
    auto whiteCount = area - cv::countNonZero(~dst);
    /*
    cv::namedWindow("bin");
    cv::imshow("bin", dst);
    cv::waitKey();
    cv::destroyAllWindows();
    //*/
    EXPECT_EQ(area, blackCount + whiteCount);
}

TEST(NiblackBinarization, MinKernelSizeTest)
{
    const auto fileName = "images/Lenna.png";
    auto src = cv::imread(fileName, 0);
    cv::Mat dst;

    ASSERT_THROW(NiblackBinarization::Binarize(src, dst, 2, -0.2), std::invalid_argument);
    ASSERT_NO_THROW(NiblackBinarization::Binarize(src, dst, 3, -0.2));
}

TEST(NiblackBinarization, ShouldSrcMatCV8UC1)
{
    const auto fileName = "images/Lenna.png";
    auto src = cv::imread(fileName, cv::IMREAD_COLOR);
    cv::Mat dst;

    ASSERT_THROW(NiblackBinarization::Binarize(src, dst, 10, -0.2), std::invalid_argument);
}