#include "PhoneDetector.hpp"
#include "AppConfig.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace {
    double AngleCosine(const cv::Point& pt1, const cv::Point& pt0, const cv::Point& pt2) {
        cv::Point v1 = pt1 - pt0;
        cv::Point v2 = pt2 - pt0;
        double n1 = cv::norm(v1);
        double n2 = cv::norm(v2);
        
        if (n1 > std::numeric_limits<double>::epsilon()) v1 /= n1;
        if (n2 > std::numeric_limits<double>::epsilon()) v2 /= n2;
        
        return std::clamp<double>(v1.dot(v2), -1.0, 1.0);
    }
}

PhoneResult PhoneDetector::DetectBestPhone(const std::vector<std::vector<cv::Point>>& contours) const {
    PhoneResult best_result;
    double max_area = 0.0;
    std::vector<cv::Point> approx;

    for (const auto& contour : contours) {

        double area = cv::contourArea(contour);
        if (area < Constants::MIN_PHONE_AREA) continue;

        double perimeter = cv::arcLength(contour, true);
        cv::approxPolyDP(contour, approx, 0.02 * perimeter, true);

        if (approx.size() == 4 && cv::isContourConvex(approx)) {
            double max_cos = GetMaxCosineDeviation(approx);
            if (max_cos < 0.2) {
                if (area > max_area) {
                    max_area = area;
                    best_result.found = true;
                    best_result.area = area;
                    best_result.rect = cv::minAreaRect(contour);
                }
            }
        }
    }

    return best_result;
}

double PhoneDetector::GetMaxCosineDeviation(const std::vector<cv::Point>& approx) const {
    double max_cos = 0.0;
    for (size_t j = 0; j < 4; j++) {
        double cos_val = std::abs(AngleCosine(approx[j], approx[(j + 1) % 4], approx[(j + 2) % 4]));
        max_cos = std::max(max_cos, cos_val);
    }
    return max_cos;
}