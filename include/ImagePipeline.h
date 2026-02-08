#pragma once
#include <opencv2/core.hpp>

class ImagePipeline {
public:
    ImagePipeline();
    cv::Mat ProcessFrame(const cv::Mat& frame);
private:
    cv::Mat _gray, _blurred, _edges, _kernel;
};