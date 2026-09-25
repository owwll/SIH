#pragma once

#include "navicore/types.hpp"
#include <vector>
#include <string>
#include <cmath>

namespace navicore {

struct RoadSegment {
    std::string id;
    double start_lat;
    double start_lon;
    double end_lat;
    double end_lon;
    float heading_rad;
    float length_m;
};

struct SnappedResult {
    double lat;
    double lon;
    std::string road_id;
    float confidence;
};

/**
 * @brief Offline Hidden Markov Model (HMM) Viterbi Map-Matching Engine.
 * Evaluates emission likelihood (orthogonal distance) and transition likelihood (road network connectivity).
 */
class HmmMapMatcher {
public:
    HmmMapMatcher(float sigma_z = 4.0f, float beta = 3.0f);

    /**
     * @brief Load road segments into local spatial index.
     */
    void LoadRoadNetwork(const std::vector<RoadSegment>& segments);

    /**
     * @brief Snaps a dead-reckoned state to the most probable road centerline.
     * @return nullopt if off-road / parking structure with no matching road.
     */
    std::optional<SnappedResult> Match(const FusionState& state);

private:
    float sigma_z_{4.0f}; // Measurement standard deviation (m)
    float beta_{3.0f};    // Transition parameter (m)
    std::vector<RoadSegment> network_;
    std::string last_matched_segment_id_{""};

    float ComputePerpendicularDistance(double lat, double lon, const RoadSegment& seg, double& snap_lat, double& snap_lon) const;
};

} // namespace navicore
