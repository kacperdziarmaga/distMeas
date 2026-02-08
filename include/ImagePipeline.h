#pragma once
#include <opencv2/core.hpp>

class ImagePipeline {
public:
    ImagePipeline();
    // Processes a raw frame and returns the binary edge map.
    cv::Mat ProcessFrame(const cv::Mat& frame);
private:
    cv::Mat _gray, _blurred, _edges, _kernel;
};