
#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>
#include "../src/NatBinarization.h"

using namespace Sansan::RD::DetectRectangles;

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

TEST(NatBinarization, ALLPixelsAre0Or255)
{
    auto fileName = "images/Lenna.png";
    auto src = cv::imread(fileName, 0);
    auto dst = src.clone();
    NatBinarization::Binarize(src, dst);

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
