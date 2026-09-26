#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include "navicore/types.hpp"

namespace navicore {

/**
 * Autonomous Dynamic Re-Routing & Mount Dislodgement Recovery Engine.
 * Monitors cross-track error during dead reckoning blackouts and triggers
 * re-routing if the vehicle turns off-route, while detecting sudden phone mount slips.
 */
class DynamicReroutingEngine {
public:
    DynamicReroutingEngine(float max_crosstrack_threshold_m = 30.0f, float dislodge_gyro_threshold_rads = 2.5f)
        : crosstrack_threshold_m_(max_crosstrack_threshold_m),
          dislodge_threshold_rads_(dislodge_gyro_threshold_rads),
          reroute_triggered_(false),
          dislodgement_detected_(false) {}

    /**
     * Evaluates current dead reckoning position against active route polyline.
     * @param current_lat Vehicle latitude in degrees.
     * @param current_lon Vehicle longitude in degrees.
     * @param route_coords Ordered vector of [lat, lon] waypoints.
     * @return Minimum orthogonal distance to the route in meters.
     */
    float ComputeCrossTrackError(double current_lat, double current_lon, const std::vector<std::pair<double, double>>& route_coords) {
        if (route_coords.size() < 2) return 0.0f;

        float min_dist_m = 1e9f;
        for (size_t i = 0; i < route_coords.size() - 1; ++i) {
            double lat1 = route_coords[i].first;
            double lon1 = route_coords[i].second;
            double lat2 = route_coords[i + 1].first;
            double lon2 = route_coords[i + 1].second;

            // Approximate equirectangular distance projection
            double d_lat = (lat2 - lat1) * 111319.5;
            double d_lon = (lon2 - lon1) * 111319.5 * std::cos(current_lat * M_PI / 180.0);
            double seg_len_sq = d_lat * d_lat + d_lon * d_lon;

            double p_lat = (current_lat - lat1) * 111319.5;
            double p_lon = (current_lon - lon1) * 111319.5 * std::cos(current_lat * M_PI / 180.0);

            double t = (seg_len_sq > 1e-6) ? std::clamp((p_lat * d_lat + p_lon * d_lon) / seg_len_sq, 0.0, 1.0) : 0.0;
            double proj_lat = lat1 * 111319.5 + t * d_lat;
            double proj_lon = lon1 * 111319.5 * std::cos(current_lat * M_PI / 180.0) + t * d_lon;

            double cur_x = current_lon * 111319.5 * std::cos(current_lat * M_PI / 180.0);
            double cur_y = current_lat * 111319.5;
            float dist = static_cast<float>(std::hypot(cur_x - proj_lon, cur_y - proj_lat));

            if (dist < min_dist_m) min_dist_m = dist;
        }

        reroute_triggered_ = (min_dist_m > crosstrack_threshold_m_);
        return min_dist_m;
    }

    /**
     * Checks if sudden high angular velocity indicates physical phone mount displacement.
     */
    bool CheckDislodgement(float gyro_x_rads, float gyro_y_rads, float gyro_z_rads) {
        float total_rot_rate = std::sqrt(gyro_x_rads * gyro_x_rads + gyro_y_rads * gyro_y_rads + gyro_z_rads * gyro_z_rads);
        dislodgement_detected_ = (total_rot_rate > dislodge_threshold_rads_);
        return dislodgement_detected_;
    }

    bool IsRerouteNeeded() const { return reroute_triggered_; }
    bool IsDislodged() const { return dislodgement_detected_; }

private:
    float crosstrack_threshold_m_;
    float dislodge_threshold_rads_;
    bool reroute_triggered_;
    bool dislodgement_detected_;
};

} // namespace navicore
