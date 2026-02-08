#include "PhoneDetector.h"
#include "AppConfig.h" // For MIN_PHONE_AREA
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>

// Anonymous namespace for internal math helpers (Internal Linkage)
// This keeps the global namespace clean.
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
        // 1. Area Filter (Fast rejection)
        double area = cv::contourArea(contour);
        if (area < Constants::MIN_PHONE_AREA) continue;

        // 2. Geometric Approximation
        double perimeter = cv::arcLength(contour, true);
        cv::approxPolyDP(contour, approx, 0.02 * perimeter, true);

        // 3. Shape Filters: Must be a convex quadrilateral
        if (approx.size() == 4 && cv::isContourConvex(approx)) {
            
            // 4. Rectangularity Filter (Cosine Check)
            // Ensures the shape is actually a rectangle, not a diamond/trapezoid
            double max_cos = GetMaxCosineDeviation(approx);

            // Threshold < 0.2 ensures angles are close to 90 degrees
            if (max_cos < 0.2) {
                // 5. Select Best (Largest)
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
    
    // Iterate through the 4 corners
    // approx has exactly 4 points here based on the calling check
    for (size_t j = 0; j < 4; j++) {
        
        // Calculate angle at p1, formed by p0-p1-p2?
        // Original code logic: angle at approx[(j+1)%4]
        // Note: The original code passed approx[j], approx[j+1], approx[j+2]. 
        // The vertex of the angle is the MIDDLE point (2nd arg).
        
        double cos_val = std::abs(AngleCosine(approx[j], approx[(j + 1) % 4], approx[(j + 2) % 4]));
        max_cos = std::max(max_cos, cos_val);
    }
    return max_cos;
}