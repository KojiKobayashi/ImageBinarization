#ifndef _NICK_BINARIZATION_H_
#define _NICK_BINARIZATION_H_
#pragma once

#include <opencv2/opencv.hpp>

namespace ImageBinarization
{
    class NickBinarization
    {
    public:
        static void Binarize(const cv::Mat &src, cv::Mat &dst, int kernelSize, double k);
        static double Binarize(const cv::Mat &src, cv::Mat &dst, int kernelSize);
    };
}

#endif // _NICK_BINARIZATION_H_
