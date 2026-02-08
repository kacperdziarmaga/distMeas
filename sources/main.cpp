#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include "AppConfig.h"
#include "ImagePipeline.h"
#include "CoinDetector.h"
#include "PhoneDetector.h"
#include "SceneRenderer.h"

int main() {
    cv::VideoCapture cap(0, cv::CAP_ANY);
    if (!cap.isOpened()) {
        std::cerr << "Error: Camera not found.\n";
        return 1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

    ImagePipeline pipeline;
    CoinDetector coin_detector;
    PhoneDetector phone_detector;
    SceneRenderer renderer;

    cv::Mat frame;

    std::cout << "Metrology System Started. Press ESC to exit.\n";

    while (true) {
        if (!cap.read(frame)) break;

        cv::Mat edges = pipeline.ProcessFrame(frame);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        CoinResult coin_res = coin_detector.DetectBestCoin(contours);
        PhoneResult phone_res = phone_detector.DetectBestPhone(contours);

        double px_per_mm = 0.0;

        if (coin_res.found) {
            double major = std::max(coin_res.rect.size.width, coin_res.rect.size.height);
            double minor = std::min(coin_res.rect.size.width, coin_res.rect.size.height);

            px_per_mm = major / Constants::COIN_REAL_DIAMETER_MM;
            double dist_mm = (Constants::COIN_REAL_DIAMETER_MM * Constants::FOCAL_LENGTH_PX) / major;
            
            double ratio = std::clamp(minor / major, 0.0, 1.0);
            double tilt_deg = std::acos(ratio) * (180.0 / CV_PI);

            renderer.RenderCoin(frame, coin_res.rect, dist_mm, tilt_deg);
        }

        if (phone_res.found) {
            double width_mm = 0.0;
            double height_mm = 0.0;

            if (coin_res.found && px_per_mm > std::numeric_limits<double>::epsilon()) {
                double s1 = phone_res.rect.size.width;
                double s2 = phone_res.rect.size.height;
                
                double short_side_px = std::min(s1, s2);
                double long_side_px  = std::max(s1, s2);

                width_mm  = short_side_px / px_per_mm;
                height_mm = long_side_px  / px_per_mm;
            }
            renderer.RenderPhone(frame, phone_res.rect, width_mm, height_mm);
        }
        renderer.RenderDebugPip(frame, edges);

        cv::imshow("Metrology Fixed", frame);
        if (cv::waitKey(1) == 27) break; // ESC
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}