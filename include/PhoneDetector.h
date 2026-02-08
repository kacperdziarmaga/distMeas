#pragma once
#include <opencv2/core.hpp>
#include <vector>

/**
 * @brief Data Transfer Object (DTO) for phone detection results.
 * adheres to SRP: purely for data transport, no logic.
 */
struct PhoneResult {
    bool found = false;
    cv::RotatedRect rect;
    double area = 0.0;
};

class PhoneDetector {
public:
    PhoneDetector() = default;

    /**
     * @brief Analyzes contours to find the best candidate for a phone.
     * 
     * Applies the following filters:
     * 1. Area > MIN_PHONE_AREA
     * 2. Polygon approximation == 4 vertices
     * 3. Convexity check
     * 4. Rectangularity check (max cosine deviation)
     * 
     * @param contours The vector of contours from the image pipeline.
     * @return PhoneResult containing the best candidate (if any).
     */
    PhoneResult DetectBestPhone(const std::vector<std::vector<cv::Point>>& contours) const;

private:
    /**
     * @brief Helper to calculate the maximum cosine of angles in a quadrilateral.
     * Used to verify that corners are roughly 90 degrees.
     */
    double GetMaxCosineDeviation(const std::vector<cv::Point>& approx) const;
};