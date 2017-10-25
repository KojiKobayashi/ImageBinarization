#ifndef _NIBLACK_BINARIZATION_H_
#define _NIBLACK_BINARIZATION_H_
#pragma once

#include <opencv2/opencv.hpp>

namespace ImageBinarization
{
    class NiblackBinarization
    {
    public:
        /// <summary>
        /// Niblack Binarization
        /// </summary>
        /// <param name="src">input 8UC1 image</param>
        /// <param name="dst">output 8UC1 image</param>
        /// <param name="kernelSize">kernel size >= 3. If this size is even, 1 is added. </param>
        /// <param name="k">Niblack parameter, -0.2 default.</param>
        /// <returns></returns>
        static void Binarize(cv::Mat &src, cv::Mat &dst, int kernelSize, double k);
    };

    inline void NiblackBinarization::Binarize(cv::Mat &src, cv::Mat &dst, int kernelSize, double k = -0.2)
    {
        if (kernelSize < 3)
            throw std::invalid_argument("kernelSize should be >= 3.");

        if (src.type() != CV_8UC1)
            throw std::invalid_argument("src type should be CV_8UC1.");

        auto margin = (kernelSize / 2);
        cv::Mat border;
        cv::copyMakeBorder(src, border, margin, margin, margin, margin, cv::BORDER_REFLECT);

        cv::Mat sum, sqsum;
        cv::integral(border, sum, sqsum);

        dst = cv::Mat(src.rows, src.cols, CV_8UC1);
        auto kernelArea = (double)(kernelSize*kernelSize);
        auto invKernelArea = 1.0 / kernelArea;

        for (auto i = 0; i < dst.rows; i++)
        {
            auto *pTl = sum.ptr<int>(i);
            auto *pTr = pTl + uint(kernelSize);
            auto *pBl = sum.ptr<int>(i + kernelSize);
            auto *pBr = pBl + uint(kernelSize);

            auto *pTl2 = sqsum.ptr<double>(i);
            auto *pTr2 = pTl2 + uint(kernelSize);
            auto *pBl2 = sqsum.ptr<double>(i + kernelSize);
            auto *pBr2 = pBl2 + uint(kernelSize);

            auto *pSrc = src.ptr<uchar>(i);
            auto *pDst = dst.ptr<uchar>(i);

            for (auto j = 0; j < dst.cols; j++)
            {
                auto sumValue = *pBr + *pTl - *pBl - *pTr;
                auto sqSumValue = *pBr2 + *pTl2 - *pBl2 - *pTr2;


                auto mean = sumValue *invKernelArea;
                auto sigma = sqSumValue * invKernelArea - mean * mean;
                sigma = sigma < 0.0 ? 0.0 : sqrt(sigma);
                auto th = mean + k * sigma;
                *pDst = double(*pSrc) > th ? 255U : 0U;

                pTl++;
                pTr++;
                pBl++;
                pBr++;
                pTl2++;
                pTr2++;
                pBl2++;
                pBr2++;
                pSrc++;
                pDst++;
            }
        }

        return;
    }
}
#endif // _NIBLACK_BINARIZATION_H_
