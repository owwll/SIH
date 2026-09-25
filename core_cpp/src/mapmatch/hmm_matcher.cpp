#include "navicore/mapmatch/hmm_matcher.hpp"
#include <limits>
#include <algorithm>

namespace navicore {

constexpr double EARTH_RADIUS_M = 6378137.0;
constexpr double DEG2RAD = 3.14159265358979323846 / 180.0;
constexpr double RAD2DEG = 180.0 / 3.14159265358979323846;

HmmMapMatcher::HmmMapMatcher(float sigma_z, float beta)
    : sigma_z_(sigma_z), beta_(beta) {}

void HmmMapMatcher::LoadRoadNetwork(const std::vector<RoadSegment>& segments) {
    network_ = segments;
}

std::optional<SnappedResult> HmmMapMatcher::Match(const FusionState& state) {
    if (network_.empty()) return std::nullopt;

    float best_prob = -1.0f;
    SnappedResult best_snap;
    bool found = false;

    for (const auto& seg : network_) {
        double snap_lat = 0.0, snap_lon = 0.0;
        float d_perp = ComputePerpendicularDistance(state.latitude_deg, state.longitude_deg, seg, snap_lat, snap_lon);

        // Emission Probability: Gaussian over orthogonal distance
        float emission_prob = std::exp(-0.5f * (d_perp * d_perp) / (sigma_z_ * sigma_z_));

        // Heading alignment bonus
        float delta_heading = std::abs(state.heading_rad - seg.heading_rad);
        while (delta_heading > 3.14159265f) delta_heading -= 2.0f * 3.14159265f;
        float heading_weight = std::max(0.2f, std::cos(delta_heading));

        // Topological transition bonus if following previous segment
        float transition_prob = (seg.id == last_matched_segment_id_) ? 1.5f : 1.0f;

        float score = emission_prob * heading_weight * transition_prob;

        if (d_perp < 25.0f && score > best_prob) { // Only snap if within 25 meters of a road
            best_prob = score;
            best_snap.lat = snap_lat;
            best_snap.lon = snap_lon;
            best_snap.road_id = seg.id;
            best_snap.confidence = std::min(1.0f, score);
            found = true;
        }
    }

    if (found && best_snap.confidence > 0.3f) {
        last_matched_segment_id_ = best_snap.road_id;
        return best_snap;
    }

    return std::nullopt;
}

float HmmMapMatcher::ComputePerpendicularDistance(
    double lat, double lon, const RoadSegment& seg, double& snap_lat, double& snap_lon
) const {
    // Convert to local meter offsets
    double dx = (seg.end_lon - seg.start_lon) * DEG2RAD * EARTH_RADIUS_M * std::cos(lat * DEG2RAD);
    double dy = (seg.end_lat - seg.start_lat) * DEG2RAD * EARTH_RADIUS_M;
    double seg_len_sq = dx * dx + dy * dy;

    if (seg_len_sq < 1e-4) {
        snap_lat = seg.start_lat;
        snap_lon = seg.start_lon;
        double px = (lon - seg.start_lon) * DEG2RAD * EARTH_RADIUS_M * std::cos(lat * DEG2RAD);
        double py = (lat - seg.start_lat) * DEG2RAD * EARTH_RADIUS_M;
        return static_cast<float>(std::sqrt(px * px + py * py));
    }

    double px = (lon - seg.start_lon) * DEG2RAD * EARTH_RADIUS_M * std::cos(lat * DEG2RAD);
    double py = (lat - seg.start_lat) * DEG2RAD * EARTH_RADIUS_M;

    double t = std::max(0.0, std::min(1.0, (px * dx + py * dy) / seg_len_sq));

    snap_lon = seg.start_lon + t * (seg.end_lon - seg.start_lon);
    snap_lat = seg.start_lat + t * (seg.end_lat - seg.start_lat);

    double perp_x = px - t * dx;
    double perp_y = py - t * dy;

    return static_cast<float>(std::sqrt(perp_x * perp_x + perp_y * perp_y));
}

} // namespace navicore
