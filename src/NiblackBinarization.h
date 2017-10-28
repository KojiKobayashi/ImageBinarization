#ifndef _NIBLACK_BINARIZATION_H_
#define _NIBLACK_BINARIZATION_H_
#pragma once

#include <opencv2/opencv.hpp>

namespace ImageBinarization
{
    class NiblackBinarization
    {
    public:
        static void Binarize(cv::Mat &src, cv::Mat &dst, int kernelSize, double k = -0.2);
    };
}
#endif // _NIBLACK_BINARIZATION_H_
