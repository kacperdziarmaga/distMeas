#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <format>
#include <numeric>

constexpr double COIN_REAL_DIAMETER_MM = 24.0, MIN_COIN_AREA = 200.0, MIN_PHONE_AREA = 1500.0; 

double angle_cosine(cv::Point pt1, cv::Point pt0, cv::Point pt2) {
    cv::Point2d v1 = cv::Point2d(pt1) - cv::Point2d(pt0);
    cv::Point2d v2 = cv::Point2d(pt2) - cv::Point2d(pt0);
    return v1.dot(v2) / sqrt(v1.dot(v1) * v2.dot(v2) + 1e-10);
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
        cv::GaussianBlur(gray, blurred, cv::Size(0, 0), 4);
        cv::Canny(blurred, edges, 5, 30, 3, true);
        cv::dilate(edges, edges, kernel, cv::Point(-1, -1), 2);

        std::vector<std::vector<cv::Point>> contours; cv::RotatedRect best_coin_rect, best_phone_rect;
        double max_coin_area = 0, max_phone_area = 0; bool coin_found = false, phone_found = false;
        std::vector<cv::Point> approx;
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area < MIN_COIN_AREA) continue;

            double perimeter = cv::arcLength(contour, true);
            
            // DOUGLAS-PEUCKER ALGORITHM
            // Approximates the contour to a polygon. 
            // Epsilon = 2% of perimeter is standard for shape counting.
            cv::approxPolyDP(contour, approx, 0.02 * perimeter, true);
            
            size_t vertices = approx.size();
            bool is_convex = cv::isContourConvex(approx);

            // --- PHONE DETECTION LOGIC ---
            // 1. Must have exactly 4 corners
            // 2. Must be convex
            // 3. Must be large
            // 4. Corners must be roughly 90 degrees (prevents random trapezoids)
            if (vertices == 4 && is_convex && area > MIN_PHONE_AREA) {
                
                double max_cos = 0;
                for (size_t j = 0; j < 4; j++) {
                    double cos_val = std::abs(angle_cosine(approx[j], approx[(j+1)%4], approx[(j+2)%4]));
                    max_cos = std::max(max_cos, cos_val);
                }

                // max_cos < 0.3 means all angles are between ~72 and 108 degrees
                if (max_cos < 0.3) {
                    // Only update if this is the largest phone found so far (Stability)
                    if (area > max_phone_area) {
                        max_phone_area = area;
                        best_phone_rect = cv::minAreaRect(contour);
                        phone_found = true;
                    }
                }
            }
            // --- COIN DETECTION LOGIC ---
            // 1. Many vertices (> 6) because circles approximate to many lines
            // 2. High circularity metric
            else if (vertices > 6) {
                double circularity = (4 * CV_PI * area) / (perimeter * perimeter);
                
                if (circularity > 0.80 && circularity < 1.2) {
                    if (area > max_coin_area && area < 50000) { // Upper limit to avoid detecting a bowl/plate
                        max_coin_area = area;
                        best_coin_rect = cv::fitEllipse(contour);
                        coin_found = true;
                    }
                }
            }
        }

        // 3. VISUALIZATION & CALCULATION
        double px_per_mm = 0.0;

        if (coin_found) {
            cv::ellipse(frame, best_coin_rect, cv::Scalar(0, 255, 255), 2);
            cv::Point2f center = best_coin_rect.center;
            
            // Use the maximum diameter to handle slight perspective tilt
            double max_diam = std::max(best_coin_rect.size.width, best_coin_rect.size.height);
            px_per_mm = max_diam / COIN_REAL_DIAMETER_MM;
            
            cv::putText(frame, "Ref Coin", center, cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0,255,255), 2);
        }

        if (phone_found) {
            cv::Point2f pts[4];
            best_phone_rect.points(pts);
            
            for(int i=0; i<4; ++i) {
                cv::line(frame, pts[i], pts[(i+1)%4], cv::Scalar(0, 255, 0), 3);
            }

            if (coin_found && px_per_mm > 0) {
                // Determine width vs height (independent of rotation)
                double side_a = std::min(best_phone_rect.size.width, best_phone_rect.size.height);
                double side_b = std::max(best_phone_rect.size.width, best_phone_rect.size.height);

                double width_mm = side_a / px_per_mm;
                double height_mm = side_b / px_per_mm;

                std::string txt = std::format("W: {:.1f}mm", width_mm);
                std::string txt2 = std::format("H: {:.1f}mm", height_mm);
                
                cv::Point2f center = best_phone_rect.center;
                cv::putText(frame, txt, center - cv::Point2f(0, 10), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
                cv::putText(frame, txt2, center + cv::Point2f(0, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
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