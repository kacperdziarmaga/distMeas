#include <opencv2/core.hpp>
double angle_cosine(cv::Point pt1, cv::Point pt0, cv::Point pt2) {
    cv::Point v1 = pt1 - pt0, v2 = pt2 - pt0;
    double n1 = cv::norm(v1), n2 = cv::norm(v2);
    if (n1 > std::numeric_limits<double>::epsilon()) v1 /= n1; if (n2 > std::numeric_limits<double>::epsilon()) v2 /= n2;
    double cos_sim = v1.dot(v2);
    return std::clamp(cos_sim, -1.0, 1.0);
}