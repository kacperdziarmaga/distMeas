#include "ImagePipeline.hpp"
#include <opencv2/imgproc.hpp>

ImagePipeline::ImagePipeline() {
    _kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
}

cv::Mat ImagePipeline::ProcessFrame(const cv::Mat& frame) {
    cv::cvtColor(frame, _gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(_gray, _blurred, cv::Size(0, 0), 2);
    cv::Canny(_blurred, _edges, 30, 100, 3, true);
    cv::dilate(_edges, _edges, _kernel, cv::Point(-1, -1));
    return _edges;
}