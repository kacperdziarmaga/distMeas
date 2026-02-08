#include "MetrologyApp.hpp"

MetrologyApp::MetrologyApp(int cameraIndex) {
    cap.open(cameraIndex, cv::CAP_ANY);
    if (!cap.isOpened()) {
        std::cerr << "Error: Camera not found on index " << cameraIndex << ".\n";
    } else {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    }
}

MetrologyApp::~MetrologyApp() {
    if (cap.isOpened()) {
        cap.release();
    }
    cv::destroyAllWindows();
}

void MetrologyApp::Run() {
    if (!cap.isOpened()) {
        std::cerr << "Cannot run application: Camera unavailable.\n";
        return;
    }

    cv::Mat frame;
    while (true) {
        if (!cap.read(frame)) {
            std::cerr << "Error: Failed to capture frame.\n";
            break;
        }

        ProcessSingleFrame(frame);

        cv::imshow("Metrology Fixed", frame);
        
        // Exit on ESC
        if (cv::waitKey(1) == 27) break;
    }
}

void MetrologyApp::ProcessSingleFrame(cv::Mat& frame) {
    // 1. Image Pre-processing
    cv::Mat edges = pipeline.ProcessFrame(frame);

    // 2. Contour Extraction
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 3. Detection
    CoinResult coin_res = coin_detector.DetectBestCoin(contours);
    PhoneResult phone_res = phone_detector.DetectBestPhone(contours);

    // 4. Math & Rendering (Refactored)
    CalculateAndRender(frame, coin_res, phone_res);

    // 5. Debug Overlay
    renderer.RenderDebugPip(frame, edges);
}

void MetrologyApp::CalculateAndRender(cv::Mat& frame, 
                                      const CoinResult& coin_res, 
                                      const PhoneResult& phone_res) {
    double px_per_mm = 0.0;

    // --- Process Reference Object (Coin) ---
    if (coin_res.found) {
        double major = std::max(coin_res.rect.size.width, coin_res.rect.size.height);
        double minor = std::min(coin_res.rect.size.width, coin_res.rect.size.height);

        // 1. Calculate Scale
        px_per_mm = major / Constants::COIN_REAL_DIAMETER_MM;
        
        // 2. Calculate Distance
        double dist_mm = (Constants::COIN_REAL_DIAMETER_MM * Constants::FOCAL_LENGTH_PX) / major;
        
        // 3. Calculate Tilt
        double ratio = std::clamp(minor / major, 0.0, 1.0);
        double tilt_deg = std::acos(ratio) * (180.0 / CV_PI);

        // 4. Render
        renderer.RenderCoin(frame, coin_res.rect, dist_mm, tilt_deg);
    }

    // --- Process Target Object (Phone) ---
    if (phone_res.found) {
        double width_mm = 0.0;
        double height_mm = 0.0;

        // Only calculate real-world dimensions if we have a valid scale reference
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
}