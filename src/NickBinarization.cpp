#include "NickBinarization.h"
#include "NatBinarization.h"

namespace ImageBinarization
{
    /// <summary>
    /// Nick Binarization
    /// </summary>
    /// <param name="src">input 8UC1 image</param>
    /// <param name="dst">output 8UC1 image</param>
    /// <param name="kernelSize">kernel size >= 3. If this size is even, 1 is added. </param>
    /// <param name="k">Niblack parameter, usually -0.2 ~ -0.1.</param>
    /// <returns></returns>
    void NickBinarization::Binarize(const cv::Mat &src, cv::Mat &dst, int kernelSize, double k)
    {
        if (kernelSize < 3)
            throw std::invalid_argument("kernelSize should be >= 3.");

        if (src.type() != CV_8UC1)
            throw std::invalid_argument("src type should be CV_8UC1.");

        auto margin = (kernelSize / 2);
        kernelSize = 2 * margin + 1;
        cv::Mat border;
        cv::copyMakeBorder(src, border, margin, margin, margin, margin, cv::BORDER_REFLECT);

        cv::Mat sum, sqsum;
        cv::integral(border, sum, sqsum);

        dst = cv::Mat(src.rows, src.cols, CV_8UC1);
        auto kernelArea = double(kernelSize*kernelSize);
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

                auto mean = sumValue * invKernelArea;
                auto sigma = (sqSumValue - mean * mean) * invKernelArea;
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

    /// <summary>
    /// Nick Binarization.k is derived automatically.
    /// </summary>
    /// <param name="src">input 8UC1 image</param>
    /// <param name="dst">output 8UC1 image</param>
    /// <param name="kernelSize">kernel size >= 3. If this size is even, 1 is added. </param>
    /// <returns>k parameter</returns>
    double NickBinarization::Binarize(const cv::Mat &src, cv::Mat &dst, int kernelSize)
    {
        if (kernelSize < 3)
            throw std::invalid_argument("kernelSize should be >= 3.");

        if (src.type() != CV_8UC1)
            throw std::invalid_argument("src type should be CV_8UC1.");

        auto margin = (kernelSize / 2);
        kernelSize = 2 * margin + 1;
        cv::Mat border;
        cv::copyMakeBorder(src, border, margin, margin, margin, margin, cv::BORDER_REFLECT);

        cv::Mat sum, sqsum;
        cv::integral(border, sum, sqsum);

        const auto minK = -0.8;
        const auto maxK = 0.2;
        auto rangeK = maxK - minK;

        // create threashold k image
        auto thImg = cv::Mat(src.size(), CV_8UC1);
        auto kernelArea = double(kernelSize*kernelSize);
        auto invKernelArea = 1.0 / kernelArea;

        for(auto i = 0; i < src.rows; i++)
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
            auto *pTh = thImg.ptr<uchar>(i);

            for (auto j = 0; j < src.cols; j++)
            {
                auto sumValue = *pBr + *pTl - *pBl - *pTr;
                auto sqSumValue = *pBr2 + *pTl2 - *pBl2 - *pTr2;

                auto mean = sumValue * invKernelArea;
                auto sigma = (sqSumValue - mean * mean) * invKernelArea;
                sigma = sigma < 0.0 ? 0.0 : sqrt(sigma);
                auto k = (double(*pSrc) - mean) / sigma;

                k = (k - minK) *255.0 / rangeK;
                k = std::max(k, 0.0);
                k = std::min(k, 255.0);
                *pTh = uchar(k);

                pTl++;
                pTr++;
                pBl++;
                pBr++;
                pTl2++;
                pTr2++;
                pBl2++;
                pBr2++;
                pSrc++;
                pTh++;
            }
        }

        auto th = NatBinarization::Binarize(thImg, dst);

        // TODO:should return niblack parameter
        return th;
    }
}