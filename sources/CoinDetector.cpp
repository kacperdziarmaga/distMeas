#include "CoinDetector.h"
#include "AppConfig.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <vector>
#include <cmath>

CoinResult CoinDetector::DetectBestCoin(const std::vector<std::vector<cv::Point>>& contours) {
    CoinResult best_result;
    best_result.found = false;
    best_result.area = 0.0;

    std::vector<cv::Point> approx;

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < Constants::MIN_COIN_AREA) continue;

        double perimeter = cv::arcLength(contour, true);
        cv::approxPolyDP(contour, approx, 0.02 * perimeter, true);
        
        size_t vertices = approx.size();
        bool is_convex = cv::isContourConvex(approx);

        // Logic Step 1: Geometric Pre-check
        if (vertices > 6 && is_convex) {
            
            // Logic Step 2: Initial Fit
            cv::RotatedRect fit = cv::fitEllipse(contour);

            // Logic Step 3: Upper Edge Filtering
            // Requirement: "Only the coin's upper edge should be considered"
            std::vector<cv::Point> upper_edge_pts;
            upper_edge_pts.reserve(contour.size());
            
            // Lambda captures 'fit' to filter points above the center
            std::copy_if(contour.begin(), contour.end(), std::back_inserter(upper_edge_pts),
                         [&](const cv::Point& p) { return p.y < fit.center.y; });

            // Refine fit if enough points exist
            if (upper_edge_pts.size() >= 6) {
                fit = cv::fitEllipse(upper_edge_pts);
            }

            // Logic Step 4: Metric Validation
            double major = std::max(fit.size.width, fit.size.height);
            double minor = std::min(fit.size.width, fit.size.height);
            
            // Avoid division by zero
            double axis_ratio = (major > std::numeric_limits<double>::epsilon()) ? minor / major : 0.0;
            double fit_area = (CV_PI * major * minor) / 4.0;

            // Logic Step 5: Selection (Perspective tilt > 0.2, Area constraints)
            // Note: 50000 is the max area hardcoded in original
            if (axis_ratio > 0.2 && fit_area > best_result.area && fit_area < 50000.0) {
                
                // We found a better coin candidate
                best_result.area = fit_area;
                best_result.rect = fit;
                best_result.found = true;

                // Logic Step 6: Homography Calculation
                std::vector<cv::Point2f> src_pts(4);
                fit.points(src_pts.data());

                // Sort corners: TL, TR, BR, BL
                std::sort(src_pts.begin(), src_pts.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
                    return a.y < b.y; 
                });
                if (src_pts[0].x > src_pts[1].x) std::swap(src_pts[0], src_pts[1]);
                if (src_pts[2].x > src_pts[3].x) std::swap(src_pts[2], src_pts[3]);

                // Destination: Square of size 'major'
                float s = static_cast<float>(major);
                std::vector<cv::Point2f> dst_pts = {
                    {0, 0}, {s, 0}, {s, s}, {0, s}
                };

                best_result.homography = cv::findHomography(src_pts, dst_pts);
            }
        }
    }

    return best_result;
}