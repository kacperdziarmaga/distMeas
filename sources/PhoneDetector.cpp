#include "PhoneDetector.hpp"
#include "GeometryUtils.hpp"
#include "AppConfig.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>

PhoneResult PhoneDetector::DetectBestPhone(const std::vector<std::vector<cv::Point>>& contours) const {
    PhoneResult best_result;
    double max_area = 0.0;

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < Constants::MIN_PHONE_AREA) continue;

        cv::RotatedRect rect = cv::minAreaRect(contour);
        double rect_area = rect.size.width * rect.size.height;

        if (rect_area > 0.0 && (area / rect_area) > 0.9) {
            if (area > max_area) {
                max_area = area;
                best_result.found = true;
                best_result.area = area;
                best_result.rect = rect;
            }
        }
    }

    return best_result;
}