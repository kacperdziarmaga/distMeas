#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <format>
#include <numeric>

constexpr double COIN_REAL_DIAMETER_MM = 24.0, FOCAL_LENGTH_PX = 800.0, MIN_COIN_AREA = 500.0, MIN_PHONE_AREA = 50000.0; 

double angle_cosine(cv::Point pt1, cv::Point pt0, cv::Point pt2) {
    cv::Point v1 = pt1 - pt0, v2 = pt2 - pt0;
    double n1 = cv::norm(v1), n2 = cv::norm(v2);
    if (n1 > std::numeric_limits<double>::epsilon()) v1 /= n1; if (n2 > std::numeric_limits<double>::epsilon()) v2 /= n2;
    double cos_sim = v1.dot(v2);
    return std::clamp(cos_sim, -1.0, 1.0);
}

int main() {
    cv::VideoCapture cap(0, cv::CAP_ANY);
    if (!cap.isOpened()) {
        std::cerr << "Error: Camera not found.\n";
        return 1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cv::Mat frame, gray, blurred, edges, kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    while (true) {
        if (!cap.read(frame)) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurred, cv::Size(0, 0), 2);
        cv::Canny(blurred, edges, 30, 100, 3, true);
        cv::dilate(edges, edges, kernel, cv::Point(-1, -1));

        std::vector<std::vector<cv::Point>> contours; cv::RotatedRect best_coin_rect, best_phone_rect;
        double max_coin_area = 0, max_phone_area = 0; bool coin_found = false, phone_found = false;
        std::vector<cv::Point> approx;
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area < MIN_COIN_AREA) continue;

            double perimeter = cv::arcLength(contour, true);
            cv::approxPolyDP(contour, approx, 0.02 * perimeter, true);
            size_t vertices = approx.size();
            bool is_convex = cv::isContourConvex(approx);

            
            //Coin Detection
            if (vertices > 6 && is_convex) {
                // 1. Initial fit to establish orientation and center
                cv::RotatedRect fit = cv::fitEllipse(contour);

                // 2. Requirement: "Only the coin's upper edge should be considered"
                // Filter points: Keep only those physically "above" the center (y < center.y)
                std::vector<cv::Point> upper_edge_pts;
                upper_edge_pts.reserve(contour.size());
                std::copy_if(contour.begin(), contour.end(), std::back_inserter(upper_edge_pts),
                             [&](const cv::Point& p) { return p.y < fit.center.y; });

                // Refine fit if we have enough points on the upper arc
                if (upper_edge_pts.size() >= 6) {
                    fit = cv::fitEllipse(upper_edge_pts);
                }

                // 3. Handle Angles/Distance:
                // Use Major Axis (invariant diameter) and Axis Ratio (perspective tilt)
                double major = std::max(fit.size.width, fit.size.height);
                double minor = std::min(fit.size.width, fit.size.height);
                double axis_ratio = (major > 0) ? minor / major : 0.0;

                // Calculate fitted area (more robust than contourArea for tilted circles)
                double fit_area = (CV_PI * major * minor) / 4.0;
                cv::Mat H;
                // Allow perspective tilt (ratio > 0.2 avoids lines/noise)
                if (axis_ratio > 0.2 && fit_area > max_coin_area && fit_area < 50000) {
                    max_coin_area = fit_area;
                    best_coin_rect = fit;
                    coin_found = true;

                    // 4. Requirement: Calculate Homography
                    // Map the detected ellipse corners to a perfect square (representing the circle from above)
                    // This matrix H can be used to rectify other points (like the phone)
                    std::vector<cv::Point2f> src_pts(4);
                    fit.points(src_pts.data());

                    // Sort corners for consistent mapping (TL, TR, BR, BL)
                    // (Simple y-sort for top/bottom, then x-sort)
                    std::sort(src_pts.begin(), src_pts.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
                        return a.y < b.y; 
                    });
                    if (src_pts[0].x > src_pts[1].x) std::swap(src_pts[0], src_pts[1]);
                    if (src_pts[2].x > src_pts[3].x) std::swap(src_pts[2], src_pts[3]);

                    // Destination: Square of size 'major' (the real diameter in pixels)
                    float s = static_cast<float>(major);
                    std::vector<cv::Point2f> dst_pts = {
                        {0, 0}, {s, 0}, {s, s}, {0, s}
                    };

                    H = cv::findHomography(src_pts, dst_pts);
                    // H is now calculated. To use it for the phone, you would apply:
                    // cv::perspectiveTransform(phone_contour, rectified_phone, H);
                }
            }
            // Phone Detection
            else if (vertices == 4 && is_convex && area > MIN_PHONE_AREA) {
                
                double max_cos = 0;
                for (size_t j = 0; j < 4; j++) {
                    double cos_val = std::abs(angle_cosine(approx[j], approx[(j+1)%4], approx[(j+2)%4]));
                    max_cos = std::max(max_cos, cos_val);
                }

                if (max_cos < 0.2) {
                    if (area > max_phone_area) {
                        max_phone_area = area;
                        best_phone_rect = cv::minAreaRect(contour);
                        phone_found = true;
                    }
                }
            }
        }


        // Visualization & Measurement
        double px_per_mm = 0.0;

        if (coin_found) {
            cv::ellipse(frame, best_coin_rect, cv::Scalar(0, 255, 255), 2);
            cv::Point2f center = best_coin_rect.center;
            
            double major = std::max(best_coin_rect.size.width, best_coin_rect.size.height);
            double minor = std::min(best_coin_rect.size.width, best_coin_rect.size.height);
            
            double tilt_deg = std::acos(std::clamp(minor / major, 0.0, 1.0)) * (180.0 / CV_PI);
            
            double dist_mm = (COIN_REAL_DIAMETER_MM * FOCAL_LENGTH_PX) / major;
            px_per_mm = major / COIN_REAL_DIAMETER_MM;
            
            cv::putText(frame, std::format("Tilt: {:.1f} deg", tilt_deg), center + cv::Point2f(0, 45), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0,255,255), 2);
            cv::putText(frame, std::format("Dist: {:.1f}mm", dist_mm), center + cv::Point2f(0, 25), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0,255,255), 2);
        }

        if (phone_found) {
            cv::Point2f pts[4];
            best_phone_rect.points(pts);
            
            for(int i=0; i<4; ++i) {
                cv::line(frame, pts[i], pts[(i+1)%4], cv::Scalar(0, 255, 0), 3);
            }

            if (coin_found && px_per_mm > std::numeric_limits<double>::epsilon()) {
                const auto [s_side, l_side] = std::minmax(best_phone_rect.size.width, best_phone_rect.size.height);
                struct Tag { double dim; std::string_view pre; float off; };
                const std::array<Tag, 2> tags {{
                    {s_side / px_per_mm, "W", -10.f},
                    {l_side / px_per_mm, "H",  25.f}
                }};
                for (const auto& t : tags) {
                    cv::putText(frame, std::format("{}: {:.1f}mm", t.pre, t.dim),
                                best_phone_rect.center + cv::Point2f(0, t.off),
                                cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 0}, 2);
                }
            }
        }

        // Debug View (Picture-in-Picture)
        cv::Mat debug_small;
        cv::resize(edges, debug_small, cv::Size(), 0.25, 0.25);
        cv::cvtColor(debug_small, debug_small, cv::COLOR_GRAY2BGR);
        debug_small.copyTo(frame(cv::Rect(0, 0, debug_small.cols, debug_small.rows)));
        cv::putText(frame, "Canny Edge", cv::Point(5, 15), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0,0,255), 1);

        cv::imshow("Metrology Fixed", frame);
        if (cv::waitKey(1) == 27) break; // ESC
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}