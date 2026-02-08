#include <opencv2/core.hpp>
struct CoinResult {
    bool found = false;
    cv::RotatedRect rect;
    cv::Mat homography;
    double area = 0.0;
};

class CoinDetector {
public:
    CoinResult DetectBestCoin(const std::vector<std::vector<cv::Point>>& contours);
};