#ifndef _NIBLACK_BINARIZATION_H_
#define _NIBLACK_BINARIZATION_H_
#pragma once

#include <opencv2/opencv.hpp>

namespace ImageBinarization
{
    class NiblackBinarization
    {
    public:
        static void Binarize(cv::Mat &src, cv::Mat &dst, double k, double kernelSize);
    };

    inline void NiblackBinarization::Binarize(cv::Mat &src, cv::Mat &dst, double k, double kernelSize)
    {
        // TODO:kernelSize is add

        cv::namedWindow("src");
        cv::imshow("src", src);

        auto margin = (kernelSize / 2);
        cv::Mat border;
        cv::copyMakeBorder(src, border, margin, margin, margin, margin, cv::BORDER_REFLECT);

        cv::Mat sum, sqsum;
        cv::integral(border, sum, sqsum);

        dst = cv::Mat(src.rows, src.cols, CV_8UC1);
        auto kernelArea = (double)(kernelSize*kernelSize);
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

                // TODO:speedup
                auto mean = sumValue / kernelArea;
                auto th = mean + k * sqrt(sqSumValue / kernelArea - mean * mean);

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
