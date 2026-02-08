#include "SceneRenderer.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp> // For fonts
#include <format> // C++20
#include <array>

void SceneRenderer::RenderCoin(cv::Mat& target, const cv::RotatedRect& rect, double dist_mm, double tilt_deg) const {
    // 1. Draw Geometry
    cv::ellipse(target, rect, _colorCoin, 2);

    // 2. Draw Text Info
    // Offsets (0, 45) and (0, 25) are layout specifics, distinct from business logic.
    cv::Point2f center = rect.center;
    
    std::string txt_tilt = std::format("Tilt: {:.1f} deg", tilt_deg);
    std::string txt_dist = std::format("Dist: {:.1f}mm", dist_mm);

    cv::putText(target, txt_tilt, center + cv::Point2f(0, 45), 
                cv::FONT_HERSHEY_PLAIN, 1.2, _colorText, 2);
    cv::putText(target, txt_dist, center + cv::Point2f(0, 25), 
                cv::FONT_HERSHEY_PLAIN, 1.2, _colorText, 2);
}

void SceneRenderer::RenderPhone(cv::Mat& target, const cv::RotatedRect& rect, double width_mm, double height_mm) const {
    // 1. Draw Geometry (Bounding Box)
    cv::Point2f pts[4];
    rect.points(pts);
    for(int i = 0; i < 4; ++i) {
        cv::line(target, pts[i], pts[(i+1)%4], _colorPhone, 3);
    }

    // 2. Draw Text Info (Dimensions)
    // We assume the caller passed width_mm/height_mm correctly.
    // If they are <= 0, we simply don't draw text (or draw 0.0mm).
    if (width_mm > 0 && height_mm > 0) {
        struct Tag { double dim; std::string_view pre; float off; };
        const std::array<Tag, 2> tags {{
            {width_mm, "W", -10.f},
            {height_mm, "H",  25.f}
        }};

        for (const auto& t : tags) {
            std::string text = std::format("{}: {:.1f}mm", t.pre, t.dim);
            cv::putText(target, text,
                        rect.center + cv::Point2f(0, t.off),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, _colorPhone, 2);
        }
    }
}

void SceneRenderer::RenderDebugPip(cv::Mat& target, const cv::Mat& edge_mask) const {
    if (edge_mask.empty()) return;

    // Create the small view
    cv::Mat debug_small;
    cv::resize(edge_mask, debug_small, cv::Size(), 0.25, 0.25);
    
    // Convert Gray -> BGR so it can be copied onto the color frame
    cv::cvtColor(debug_small, debug_small, cv::COLOR_GRAY2BGR);
    
    // Copy to Top-Left corner
    // Ensure we don't go out of bounds
    if (debug_small.cols <= target.cols && debug_small.rows <= target.rows) {
        debug_small.copyTo(target(cv::Rect(0, 0, debug_small.cols, debug_small.rows)));
    }

    cv::putText(target, "Canny Edge", cv::Point(5, 15), 
                cv::FONT_HERSHEY_PLAIN, 1.0, _colorDebug, 1);
}