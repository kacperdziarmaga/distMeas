#ifndef METROLOGY_APP_HPP
#define METROLOGY_APP_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include "AppConfig.hpp"
#include "ImagePipeline.hpp"
#include "CoinDetector.hpp"
#include "PhoneDetector.hpp"
#include "SceneRenderer.hpp"

class MetrologyApp {
public:
    /**
     * @brief Construct a new Metrology App object
     * @param cameraIndex The ID of the camera to open (default 0)
     */
    explicit MetrologyApp(int cameraIndex = 0);

    /**
     * @brief Destructor to ensure resources are released
     */
    ~MetrologyApp();

    /**
     * @brief Starts the main processing loop.
     */
    void Run();

private:
    cv::VideoCapture cap;
    ImagePipeline pipeline;
    CoinDetector coin_detector;
    PhoneDetector phone_detector;
    SceneRenderer renderer;

    double px_per_mm;

    void CalculateAndRender(cv::Mat& frame, const CoinResult& coin_res, const PhoneResult& phone_res);
    void ProcessSingleFrame(cv::Mat& frame);
};

#endif
