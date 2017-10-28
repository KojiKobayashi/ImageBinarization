#ifndef _NAT_BINARIZATION_H_
#define  _NAT_BINARIZATION_H_
#pragma once

#include <opencv2/opencv.hpp>

namespace ImageBinarization
{
    class NatBinarization
    {
    public:
        static void Binarize(const cv::Mat &src, cv::Mat &dst);

    protected:
        static void CreateLabelCountHistogram(const cv::Mat& src, int(&hist)[256]);

    private:
        static int GetBaseLabel(int label, std::map<int, int> &baseLabels);
        static int OtshToHistogram(int(&hist)[256]);
    };
}

#endif // _NAT_BINARIZATION_H_
