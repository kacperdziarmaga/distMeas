#include <opencv2/opencv.hpp>
#include <print>
#include <vector>
#include <cmath>
#include <algorithm>
#include <execution>

constexpr double REAL_DIAMETER_MM = 24.0;
constexpr double FOCAL_LENGTH_PX = 85.0; 

int main() {
    cv::setUseOptimized(true);
    cv::setNumThreads(cv::getNumberOfCPUs());

    cv::VideoCapture cap(0, cv::CAP_ANY);
    if (!cap.isOpened()) {
        std::println(stderr, "Error: Could not open camera.");
        return 1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(cv::CAP_PROP_FPS, 60);

    cv::Mat frame, gray, blurred;
    std::vector<cv::Vec3f> circles;
    circles.reserve(100);

    cv::TickMeter tm;

    while (true) {
        tm.start();
        
        if (!cap.read(frame)) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurred, cv::Size(9, 9), 2, 2);

        circles.clear();
        cv::HoughCircles(blurred, circles, cv::HOUGH_GRADIENT, 
                         1.0,               
                         blurred.rows / 8,  
                         100.0,             
                         30.0,              
                         20,                
                         400                
        );

        if (!circles.empty()) {
            const auto& c = circles[0];
            cv::Point center(cvRound(c[0]), cvRound(c[1]));
            double radius = c[2];
            double diameter_px = radius * 2.0;

            if (diameter_px > 0) {
                double distance_mm = (REAL_DIAMETER_MM * FOCAL_LENGTH_PX) / diameter_px;

                cv::circle(frame, center, 3, cv::Scalar(0, 255, 0), -1);
                cv::circle(frame, center, cvRound(radius), cv::Scalar(0, 0, 255), 2);
                
                std::string dist_text = std::format("Dist: {:.2f} mm", distance_mm);
                cv::putText(frame, dist_text, center + cv::Point(0, -20), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            }
        }

        tm.stop();
        double fps = tm.getFPS();
        tm.reset();

        cv::putText(frame, std::format("FPS: {:.1f}", fps), cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 0), 2);

        cv::imshow("5 PLN Distance Measurement", frame);

        if (cv::waitKey(1) == 27) break; 
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}