
#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>
#include "../src/NatBinarization.h"

using namespace ImageBinarization;

class NatBinarizationForTest:NatBinarization
{
public:
    /// <summary>
    /// ヒストグラム作成箇所テスト用
    /// </summary>
    /// <param name="src"></param>
    /// <param name="hist"></param>
    static void DriveCreateLabelCountHistogram(cv::Mat &src, int(&hist)[256])
    {
        CreateLabelCountHistogram(src, hist);
    }
};

TEST(NatBinarization, CreateLabelCountHistogramTest)
{
    cv::Mat src = cv::Mat::zeros(cv::Size(5, 5), CV_8UC1);
    int hist[256];

    src.at<uchar>(2, 2) = 100;
    src.at<uchar>(2, 1) = 100;
    src.at<uchar>(1, 2) = 100;
    src.at<uchar>(2, 3) = 100;
    src.at<uchar>(3, 2) = 100;

    NatBinarizationForTest::DriveCreateLabelCountHistogram(src, hist);
    EXPECT_EQ(1, hist[0]);
    EXPECT_EQ(1, hist[100]);
    EXPECT_EQ(1, hist[255]);

    src = 255;
    src.at<uchar>(2, 2) = 101;
    src.at<uchar>(2, 1) = 100;
    src.at<uchar>(1, 2) = 100;
    src.at<uchar>(2, 3) = 100;
    src.at<uchar>(3, 2) = 100;

    NatBinarizationForTest::DriveCreateLabelCountHistogram(src, hist);
    EXPECT_EQ(0, hist[0]);
    EXPECT_EQ(4, hist[100]);
    EXPECT_EQ(1, hist[101]);
    EXPECT_EQ(1, hist[255]);
}

TEST(NatBinarization, NormalTest)
{
    const auto fileName = "images/Lenna.png";
    auto src = cv::imread(fileName, 0);
    cv::Mat dst;
    NatBinarization::Binarize(src, dst);

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

TEST(NatBinarization, ErrCaseTest)
{
    cv::Mat src, dst;
    const auto fileName = "images/Lenna.png";

    // 8UC1 only
    src = cv::imread(fileName, cv::IMREAD_COLOR);
    ASSERT_THROW(NatBinarization::Binarize(src, dst), std::invalid_argument);
}
