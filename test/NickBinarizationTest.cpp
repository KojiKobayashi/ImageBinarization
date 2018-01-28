#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>
#include "../src/NickBinarization.h"

using namespace ImageBinarization;

TEST(NickBinarization, NormalTest)
{
    const auto fileName = "images/Lenna.png";
    auto src = cv::imread(fileName, 0);
    cv::Mat dst;
    NickBinarization::Binarize(src, dst, 10, -0.14);

    /*
    cv::namedWindow("bin");
    cv::imshow("bin", dst);
    cv::waitKey();
    cv::destroyAllWindows();
    //*/

    ASSERT_EQ(src.rows, dst.rows);
    ASSERT_EQ(src.cols, dst.cols);
    ASSERT_EQ(CV_8UC1, dst.type());

    auto area = src.rows * src.cols;
    auto blackCount = area - cv::countNonZero(dst);
    auto whiteCount = area - cv::countNonZero(~dst);
    EXPECT_EQ(area, blackCount + whiteCount);
}

TEST(NickBinarization, MinKernelSizeTest)
{
    const auto fileName = "images/Lenna.png";
    auto src = cv::imread(fileName, 0);
    cv::Mat dst;

    ASSERT_THROW(NickBinarization::Binarize(src, dst, 2, -0.14), std::invalid_argument);
    ASSERT_NO_THROW(NickBinarization::Binarize(src, dst, 3, -0.14));
}

TEST(NickBinarization, ShouldSrcMatCV8UC1)
{
    const auto fileName = "images/Lenna.png";
    auto src = cv::imread(fileName, cv::IMREAD_COLOR);
    cv::Mat dst;

    ASSERT_THROW(NickBinarization::Binarize(src, dst, 10, -0.14), std::invalid_argument);
}

TEST(NickBinarization, NormalTestNonPara)
{
    const auto fileName = "images/Lenna.png";
    auto src = cv::imread(fileName, 0);
    cv::Mat dst;
    auto k = NickBinarization::Binarize(src, dst, 10);

    /*
    cv::namedWindow("bin");
    cv::imshow("bin", dst);
    cv::waitKey();
    cv::destroyAllWindows();
    //*/

    ASSERT_EQ(src.rows, dst.rows);
    ASSERT_EQ(src.cols, dst.cols);
    ASSERT_EQ(CV_8UC1, dst.type());

    auto area = src.rows * src.cols;
    auto blackCount = area - cv::countNonZero(dst);
    auto whiteCount = area - cv::countNonZero(~dst);
    EXPECT_EQ(area, blackCount + whiteCount);
}