#include "NatBinarization.h"

namespace ImageBinarization
{
    /// <summary>
    /// NatBinarization Binarization
    /// </summary>
    /// <param name="src">input 8UC1 image</param>
    /// <param name="dst">output 8UC1 image</param>
    /// <returns></returns>
    void NatBinarization::Binarize(const cv::Mat &src, cv::Mat &dst)
    {
        if (src.type() != CV_8UC1)
            throw std::invalid_argument("src should be 8CU1.");

        int hist[256];
        CreateLabelCountHistogram(src, hist);

        auto max = 0;
        for (auto i = 0; i < 256; i++)
        {
            if (hist[i] > max)
                max = hist[i];
        }
        for (auto i = 0; i < 256; i++)
        {
            hist[i] = (hist[i] * 255) / max;
        }

        auto th = OtshToHistogram(hist);
        cv::threshold(src, dst, th, 255, CV_THRESH_BINARY);
        return;
    }

    void NatBinarization::CreateLabelCountHistogram(const cv::Mat& src, int(&hist)[256])
    {
        // bin map �쐬
        // <0:end point, >=0:next pixel position with same bin
        int startPtr[256];
        int endPtr[256];
        for (auto i = 0; i < 256; i++)
        {
            startPtr[i] = -1;
            endPtr[i] = -1;
        }

        cv::Mat binMap = cv::Mat::zeros(src.rows, src.cols, CV_32SC1);
        binMap = -1;

        for (auto i = 0; i < src.rows; i++)
        {
            auto *pSrc = src.ptr(i);
            auto yPos = i << 15;

            for (auto j = 0; j < src.cols; j++)
            {
                auto newPos = (yPos | j);
                auto val = *pSrc;

                if (endPtr[val] >= 0)
                {
                    auto x = endPtr[val] & 0x7FFF;
                    auto y = endPtr[val] >> 15;
                    binMap.at<int>(y, x) = newPos;
                    endPtr[val] = newPos;
                }
                else
                {
                    startPtr[val] = newPos;
                    endPtr[val] = newPos;
                }

                pSrc++;
            }
        }

        // 0:not labelid yet, <0:baselabel(labelIndex * -1), >0:refrrence label Index
        cv::Mat regionMap = cv::Mat::zeros(src.rows + 2, src.cols + 2, CV_32SC1);

        // <0:base(key * -1), >0:reffernce index. Empty at index 0.
        std::map<int, int> baseLabels;

        auto tmpLabel = 1;
        auto labelCount = 0;

        for (auto bin = 0; bin < 256; bin++)
        {
            if (startPtr[bin] < 0)
            {
                hist[bin] = labelCount;
                continue;
            }

            auto pos = startPtr[bin];
            while (pos >= 0)
            {
                auto x = pos & 0x7FFF;
                auto y = pos >> 15;

                auto uLabel = regionMap.at<int>(y, x + 1);
                auto lLabel = regionMap.at<int>(y + 1, x);
                auto rLabel = regionMap.at<int>(y + 1, x + 2);
                auto bLabel = regionMap.at<int>(y + 2, x + 1);

                // �Ɨ��_�͐V�K���x�����s
                if (uLabel == 0 && lLabel == 0 && rLabel == 0 && bLabel == 0)
                {
                    hist[bin]++;
                    baseLabels[tmpLabel] = -1 * tmpLabel; // base
                    regionMap.at<int>(y + 1, x + 1) = tmpLabel;
                    tmpLabel++;
                    labelCount++;
                    pos = binMap.at<int>(y, x);
                    continue;
                }

                auto uBaseLabel = uLabel == 0 ?
                    0 : GetBaseLabel(uLabel, baseLabels);
                auto lBaseLabel = lLabel == 0 ?
                    0 : GetBaseLabel(lLabel, baseLabels);
                auto rBaseLabel = rLabel == 0 ?
                    0 : GetBaseLabel(rLabel, baseLabels);
                auto bBaseLabel = bLabel == 0 ?
                    0 : GetBaseLabel(bLabel, baseLabels);

                std::vector<int> labels = { uBaseLabel, lBaseLabel, rBaseLabel, bBaseLabel };

                // �d���폜
                std::sort(labels.begin(), labels.end());
                labels.erase(std::unique(labels.begin(), labels.end()), labels.end());

                // �g�[�^�����x�����̍팸��
                auto delCount = labels.size() - 1;

                auto itr = labels.begin();

                // �����x������f�X�L�b�v
                if (*itr == 0)
                {
                    delCount--;
                    ++itr;
                }

                // �e�[�u����resolving
                auto setLabel = *itr;
                ++itr;
                for (; itr != labels.end(); ++itr)
                {
                    baseLabels[*itr] = setLabel;
                }

                regionMap.at<int>(y + 1, x + 1) = setLabel;

                // �������̂��߂Ƀ��x�����蒼��
                if (uLabel != 0)
                    regionMap.at<int>(y, x + 1) = setLabel;
                if (lLabel != 0)
                    regionMap.at<int>(y + 1, x) = setLabel;
                if (rLabel != 0)
                    regionMap.at<int>(y + 1, x + 2) = setLabel;
                if (bLabel != 0)
                    regionMap.at<int>(y + 2, x + 1) = setLabel;

                labelCount -= delCount;

                pos = binMap.at<int>(y, x);
            }

            hist[bin] = labelCount;
        }
    }

    int NatBinarization::GetBaseLabel(const int label, std::map<int, int> &baseLabels)
    {
        auto baseLabel1 = label;
        while (baseLabel1 >= 0)
        {
            baseLabel1 = baseLabels[baseLabel1];
        }

        return -1 * baseLabel1;
    }

    int NatBinarization::OtshToHistogram(int(&hist)[256])
    {
        auto total = 0;
        long long int sumWeight = 0;
        for (auto i = 0; i < 256; i++)
        {
            total += hist[i];
            sumWeight += i * hist[i];
        }

        auto max = 0.0;
        auto th = 0;
        auto countB = 0, countF = 0;
        long long int weightB = 0;
        for (auto i = 0; i < 256; i++)
        {
            countB += hist[i];
            if (countB == 0)
                continue;

            countF = total - countB;
            if (countF == 0)
                return th;

            weightB += i * hist[i];

            /*
            double mB = (double)weightB / countB;
            double mF = (double)(sumWeight - weightB) / countF;
            double between = countB * countF *(mB - mF)*(mB - mF);
            //*/
            ///*
            auto diff = (double)(weightB*countF - (sumWeight - weightB)*countB);
            auto between = countB * countF * diff * diff;
            //*/
            if (between > max)
            {
                max = between;
                th = i;
            }
        }

        return  th;
    }
}
