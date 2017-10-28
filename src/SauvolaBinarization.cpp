#include "SauvolaBinarization.h"

namespace ImageBinarization
{
    void SauvolaBinarization::Binarize(const cv::Mat &src, cv::Mat &dst, int kernelSize, double k, double r)
    {
        if (kernelSize < 3)
            throw std::invalid_argument("kernelSize should be >= 3.");

        if (src.type() != CV_8UC1)
            throw std::invalid_argument("src type should be CV_8UC1.");

        if (!(r > 0.0))
            throw std::invalid_argument("r should be > 0.0.");

        auto margin = (kernelSize / 2);
        kernelSize = 2 * margin + 1;
        cv::Mat border;
        cv::copyMakeBorder(src, border, margin, margin, margin, margin, cv::BORDER_REFLECT);

        cv::Mat sum, sqsum;
        cv::integral(border, sum, sqsum);

        dst = cv::Mat(src.rows, src.cols, CV_8UC1);
        auto kernelArea = (double)(kernelSize*kernelSize);
        auto invKernelArea = 1.0 / kernelArea;
        auto invR = 1.0 / r;

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

                auto th = mean *(1.0 + k * (sigma * invR - 1.0));
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