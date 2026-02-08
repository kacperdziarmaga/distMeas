#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// Module Headers
#include "AppConfig.h"       // Configuration Constants
#include "ImagePipeline.h"   // Image Processing (Gray -> Canny)
#include "CoinDetector.h"    // Coin Logic
#include "PhoneDetector.h"   // Phone Logic
#include "SceneRenderer.h"   // Visualization

int main() {
    // 1. Hardware Initialization
    cv::VideoCapture cap(0, cv::CAP_ANY);
    if (!cap.isOpened()) {
        std::cerr << "Error: Camera not found.\n";
        return 1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

    // 2. Component Instantiation
    // Dependencies are created here and exist for the application lifetime.
    ImagePipeline pipeline;
    CoinDetector coin_detector;
    PhoneDetector phone_detector;
    SceneRenderer renderer;

    cv::Mat frame;

    std::cout << "Metrology System Started. Press ESC to exit.\n";

    while (true) {
        // 3. Capture
        if (!cap.read(frame)) break;

        // 4. Preprocessing (Pixel Domain)
        // Delegate low-level image ops to the pipeline
        cv::Mat edges = pipeline.ProcessFrame(frame);

        // 5. Data Adaptation (Bridge)
        // Convert Pixel Data (Edges) -> Geometric Data (Contours)
        // This is done once per frame to serve all detectors.
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 6. Detection (Vector Domain)
        // Ask specific experts to find their objects
        CoinResult coin_res = coin_detector.DetectBestCoin(contours);
        PhoneResult phone_res = phone_detector.DetectBestPhone(contours);

        // 7. Business Logic (Metrology / Physics)
        // Main connects the dots: "If I have a coin, I know the scale."
        double px_per_mm = 0.0;

        if (coin_res.found) {
            // Extract geometry
            double major = std::max(coin_res.rect.size.width, coin_res.rect.size.height);
            double minor = std::min(coin_res.rect.size.width, coin_res.rect.size.height);

            // Calculate Physics (Scale, Distance, Tilt)
            px_per_mm = major / Constants::COIN_REAL_DIAMETER_MM;
            double dist_mm = (Constants::COIN_REAL_DIAMETER_MM * Constants::FOCAL_LENGTH_PX) / major;
            
            // Calculate Tilt (clamped for safety)
            double ratio = std::clamp(minor / major, 0.0, 1.0);
            double tilt_deg = std::acos(ratio) * (180.0 / CV_PI);

            // Visualize Coin
            renderer.RenderCoin(frame, coin_res.rect, dist_mm, tilt_deg);
        }

        if (phone_res.found) {
            double width_mm = 0.0;
            double height_mm = 0.0;

            // Only calculate dimensions if we have a valid scale
            if (coin_res.found && px_per_mm > std::numeric_limits<double>::epsilon()) {
                // Logic from original code: Short side is 'W', Long side is 'H'
                double s1 = phone_res.rect.size.width;
                double s2 = phone_res.rect.size.height;
                
                double short_side_px = std::min(s1, s2);
                double long_side_px  = std::max(s1, s2);

                width_mm  = short_side_px / px_per_mm;
                height_mm = long_side_px  / px_per_mm;
            }

            // Visualize Phone
            renderer.RenderPhone(frame, phone_res.rect, width_mm, height_mm);
        }

        // 8. Debug Visualization (Picture-in-Picture)
        renderer.RenderDebugPip(frame, edges);

        // 9. Display
        cv::imshow("Metrology Fixed", frame);
        if (cv::waitKey(1) == 27) break; // ESC
    }

    // Cleanup
    cap.release();
    cv::destroyAllWindows();
    return 0;
}