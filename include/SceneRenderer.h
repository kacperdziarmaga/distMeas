#pragma once
#include <opencv2/core.hpp>
#include <string_view>

class SceneRenderer {
public:
    SceneRenderer() = default;

    /**
     * @brief Draws the coin ellipse and overlay text.
     * @param target The frame to draw on (modified in place).
     * @param rect The fitted ellipse of the coin.
     * @param dist_mm Calculated distance (Business Logic).
     * @param tilt_deg Calculated tilt (Business Logic).
     */
    void RenderCoin(cv::Mat& target, const cv::RotatedRect& rect, double dist_mm, double tilt_deg) const;

    /**
     * @brief Draws the phone bounding box and dimensions.
     * @param target The frame to draw on.
     * @param rect The minimum area rect of the phone.
     * @param width_mm The physical width (calculated by Main).
     * @param height_mm The physical height (calculated by Main).
     */
    void RenderPhone(cv::Mat& target, const cv::RotatedRect& rect, double width_mm, double height_mm) const;

    /**
     * @brief Draws the "Picture-in-Picture" debug view of the edges.
     * @param target The main frame.
     * @param edge_mask The binary edge map from ImagePipeline.
     */
    void RenderDebugPip(cv::Mat& target, const cv::Mat& edge_mask) const;

private:
    // Constants for consistent styling (Yellow for Coin, Green for Phone)
    const cv::Scalar _colorCoin = cv::Scalar(0, 255, 255);
    const cv::Scalar _colorPhone = cv::Scalar(0, 255, 0);
    const cv::Scalar _colorText = cv::Scalar(0, 255, 255);
    const cv::Scalar _colorDebug = cv::Scalar(0, 0, 255);
};