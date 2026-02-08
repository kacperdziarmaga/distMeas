#include "MetrologyApp.hpp"

MetrologyApp::MetrologyApp(int cameraIndex) : px_per_mm(0.0) {
    cap.open(cameraIndex, cv::CAP_ANY);
    if (!cap.isOpened()) {std::cerr << "Error: Camera not found on index " << cameraIndex << ".\n";}
    else {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, Constants::CAM_RES_X);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, Constants::CAM_RES_Y);
    }
}

MetrologyApp::~MetrologyApp() {
    if (cap.isOpened()) {
        cap.release();
    }
    cv::destroyAllWindows();
}

void MetrologyApp::Run() {
    if (!cap.isOpened()) {std::cerr << "Cannot run application: Camera unavailable.\n"; return;}

    cv::Mat frame;
    while (true) {
        if (!cap.read(frame)) {std::cerr << "Error: Failed to capture frame.\n"; break;}
        ProcessSingleFrame(frame);
        cv::imshow("Metrology Fixed", frame);
        if (cv::waitKey(1) == 27) break; //ESC
    }
}

void MetrologyApp::ProcessSingleFrame(cv::Mat& frame) {

    cv::Mat edges = pipeline.ProcessFrame(frame);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    CoinResult coin_res = coin_detector.DetectBestCoin(contours);
    PhoneResult phone_res = phone_detector.DetectBestPhone(contours);

    CalculateAndRender(frame, coin_res, phone_res);

    renderer.RenderDebugPip(frame, edges);
}

void MetrologyApp::CalculateAndRender(cv::Mat& frame, const CoinResult& coin_res, const PhoneResult& phone_res) {;
    // --- Process Reference Object (Coin) ---
    if (coin_res.found) {
        auto [minor, major] = std::minmax(coin_res.rect.size.width, coin_res.rect.size.height);
        px_per_mm = major / Constants::COIN_REAL_DIAMETER_MM;
        double dist_mm = (Constants::COIN_REAL_DIAMETER_MM * Constants::FOCAL_LENGTH_PX) / major,
        tilt_deg = std::acos(std::clamp<double>(minor / major, 0.0, 1.0)) * (180.0 / CV_PI);

        renderer.RenderCoin(frame, coin_res.rect, dist_mm, tilt_deg);
    }

    // --- Process Reference Object (Phone) ---
    if (phone_res.found) {
        double width_mm = 0.0,  height_mm = 0.0;

        if (coin_res.found && px_per_mm > std::numeric_limits<double>::epsilon()) {
            double s1 = phone_res.rect.size.width,  s2 = phone_res.rect.size.height;
            auto [short_side_px, long_side_px] = std::minmax(s1, s2);
            std::tie(width_mm, height_mm) = std::make_pair(short_side_px / px_per_mm, long_side_px / px_per_mm);
        }
        renderer.RenderPhone(frame, phone_res.rect, width_mm, height_mm);
    }
}