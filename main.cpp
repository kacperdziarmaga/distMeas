#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <print> // C++23. Use <iostream> if on older compilers.

// --- CONFIGURATION ---
constexpr double REAL_DIAMETER_MM = 24.0; 
constexpr double FOCAL_LENGTH_PX = 800.0; 
constexpr double MIN_AREA = 1000.0;      // Noise filter
constexpr double MAX_ECCENTRICITY = 0.95; // Reject lines/extreme angles

// Helper to order points for perspective transform
// Orders: Top-Left, Top-Right, Bottom-Right, Bottom-Left
std::vector<cv::Point2f> order_points(const std::vector<cv::Point2f>& pts) {
    std::vector<cv::Point2f> ordered(4);
    std::vector<float> sum(4), diff(4);

    for (size_t i = 0; i < 4; ++i) {
        sum[i] = pts[i].x + pts[i].y;
        diff[i] = pts[i].y - pts[i].x;
    }

    ordered[0] = pts[std::min_element(sum.begin(), sum.end()) - sum.begin()];      // TL
    ordered[2] = pts[std::max_element(sum.begin(), sum.end()) - sum.begin()];      // BR
    ordered[1] = pts[std::min_element(diff.begin(), diff.end()) - diff.begin()];   // TR
    ordered[3] = pts[std::max_element(diff.begin(), diff.end()) - diff.begin()];   // BL
    return ordered;
}

int main() {
    // Optimization flags
    cv::setUseOptimized(true);
    cv::setNumThreads(cv::getNumberOfCPUs());

    cv::VideoCapture cap(0, cv::CAP_ANY);
    if (!cap.isOpened()) {
        std::println(stderr, "Error: Could not open camera.");
        return 1;
    }

    // High resolution helps with edge fitting accuracy
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(cv::CAP_PROP_FPS, 60);

    cv::Mat frame, gray, blurred, edges, rectified_view;
    cv::TickMeter tm;

    std::println("Press 'ESC' to exit.");

    while (true) {
        tm.start();
        if (!cap.read(frame)) break;

        // 1. Preprocessing for Contour Detection
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        // Bilateral filter preserves edges better than Gaussian for shape fitting
        cv::bilateralFilter(gray, blurred, 9, 75, 75);
        
        // Canny Edge Detection (Adaptive thresholds or tuned constants)
        // Using Otsu's threshold high value for Canny is a common heuristic
        double high_thresh = cv::threshold(blurred, edges, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        cv::Canny(blurred, edges, 0.5 * high_thresh, high_thresh);
        
        // Close gaps in edges
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::RotatedRect best_ellipse;
        double max_area = 0;
        bool found = false;

        for (const auto& contour : contours) {
            if (contour.size() < 5) continue; // fitEllipse requires >= 5 points
            
            double area = cv::contourArea(contour);
            if (area < MIN_AREA) continue;

            cv::RotatedRect rbox = cv::fitEllipse(contour);
            
            // Filter by aspect ratio (eccentricity)
            // Coin cannot look like a flat line unless at 90 deg (invisible)
            double minor = std::min(rbox.size.width, rbox.size.height);
            double major = std::max(rbox.size.width, rbox.size.height);
            if (minor / major < (1.0 - MAX_ECCENTRICITY)) continue;

            // Pick the largest valid ellipse (assumption: coin is main object)
            if (area > max_area) {
                max_area = area;
                best_ellipse = rbox;
                found = true;
            }
        }

        if (found) {
            // 2. Visualization & Distance
            cv::ellipse(frame, best_ellipse, cv::Scalar(0, 0, 255), 2);
            cv::drawMarker(frame, best_ellipse.center, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 20, 2);

            // Use Major Axis for Distance (less affected by tilt)
            double major_axis_px = std::max(best_ellipse.size.width, best_ellipse.size.height);
            double minor_axis_px = std::min(best_ellipse.size.width, best_ellipse.size.height);
            
            double distance_mm = (REAL_DIAMETER_MM * FOCAL_LENGTH_PX) / major_axis_px;
            
            // Calculate inclination angle (approximate)
            // cos(angle) = minor / major
            double inclination_deg = std::acos(minor_axis_px / major_axis_px) * (180.0 / CV_PI);

            std::string d_txt = std::format("Dist: {:.1f} mm", distance_mm);
            std::string a_txt = std::format("Tilt: {:.1f} deg", inclination_deg);

            cv::putText(frame, d_txt, best_ellipse.center + cv::Point2f(20, -10), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, a_txt, best_ellipse.center + cv::Point2f(20, 20), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);

            // 3. Rectification (Warp Plane)
            // Get 4 corners of the RotatedRect
            cv::Point2f src_pts[4];
            best_ellipse.points(src_pts);
            std::vector<cv::Point2f> src_vec(src_pts, src_pts + 4);
            src_vec = order_points(src_vec);

            // Define destination square (un-distorted view)
            // We map the elliptical bounds to a square of size major_axis x major_axis
            float size = static_cast<float>(major_axis_px);
            std::vector<cv::Point2f> dst_vec = {
                {0, 0},
                {size, 0},
                {size, size},
                {0, size}
            };

            // Calculate Homography and Warp
            cv::Mat M = cv::getPerspectiveTransform(src_vec, dst_vec);
            cv::warpPerspective(frame, rectified_view, M, cv::Size((int)size, (int)size));
            
            cv::imshow("Rectified Coin", rectified_view);
        }

        tm.stop();
        cv::putText(frame, std::format("FPS: {:.1f}", tm.getFPS()), cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 0), 2);
        tm.reset();

        cv::imshow("Angled Detection", frame);
        // Show edges for debug if needed
        // cv::imshow("Edges", edges);

        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}