#ifndef _SAUVOLA_BINARIZATION_H_
#define _SAUVOLA_BINARIZATION_H_
#pragma once

#include <opencv2/opencv.hpp>

namespace ImageBinarization
{
    class SauvolaBinarization
    {
    public:
        static void Binarize(cv::Mat &src, cv::Mat &dst, int kernelSize, double k, double r);
    }

    inline void SauvolaBinarization::Binarize(cv::Mat &src, cv::Mat &dst, int kernelSize, double k = -0.2, double r = 128)
    {
    }
}

#endif // _SAUVOLA_BINARIZATION_H_
