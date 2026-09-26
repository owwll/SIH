#include "navicore/pipeline/navicore_pipeline.hpp"

namespace navicore {

NavicorePipeline::NavicorePipeline(VehicleType vehicle_type)
    : profile_manager_(),
      current_profile_(profile_manager_.GetProfile(vehicle_type)) {
    // Configure ESKF with vehicle kinematic profile (NHC stiffness and idle band)
    eskf_.SetVehicleProfile(current_profile_);

    // Active route is initialized as empty stub (routing UI integration follow-up per TRD §6.1)
    active_route_.clear();
}

void NavicorePipeline::SetVehicleType(VehicleType type) {
    current_profile_ = profile_manager_.GetProfile(type);
    eskf_.SetVehicleProfile(current_profile_);
}

void NavicorePipeline::SetActiveRoute(const std::vector<std::pair<double, double>>& route_coords) {
    active_route_ = route_coords;
}

void NavicorePipeline::SetRoadNetwork(const std::vector<RoadSegment>& segments) {
    map_matcher_.LoadRoadNetwork(segments);
}

void NavicorePipeline::ProcessImu(const ImuSample& raw_imu, float dt_seconds) {
    // 1. Ingest raw IMU into dynamic auto-calibration engine
    calibrator_.Ingest(raw_imu);

    // 2. Transform body frame acceleration to vehicle frame if calibrated
    ImuSample veh_imu = raw_imu;
    auto rot = calibrator_.CurrentRotation();
    if (rot.has_value()) {
        float vx = 0.0f, vy = 0.0f, vz = 0.0f;
        calibrator_.TransformBodyToVehicle(raw_imu.ax, raw_imu.ay, raw_imu.az, vx, vy, vz);
        veh_imu.ax = vx;
        veh_imu.ay = vy;
        veh_imu.az = vz;
    }

    // 3. Propagate 15-state ESKF with vehicle-frame IMU sample
    eskf_.Predict(veh_imu, dt_seconds);

    // 4. Feed angular velocity to dynamic rerouting engine for mount dislodgement detection
    rerouting_engine_.CheckDislodgement(raw_imu.gx, raw_imu.gy, raw_imu.gz);
}

void NavicorePipeline::UpdateGnss(const GnssFix& fix) {
    eskf_.UpdateGnss(fix);
}

void NavicorePipeline::UpdateAiOdometer(const OdometerOutput& ai_odometry, float heading_rad) {
    eskf_.UpdateAiOdometer(ai_odometry, heading_rad);
}

PipelineOutput NavicorePipeline::GetState() {
    PipelineOutput output;

    // 1. Retrieve fused nominal state from ESKF
    output.state = eskf_.GetState();

    // 2. Wire HmmMapMatcher: after every GetState() call, match against road network
    // Returns nullopt for off-road areas (e.g. multi-floor parking basements)
    last_snapped_road_ = map_matcher_.Match(output.state);
    output.snapped_road = last_snapped_road_;

    // 3. Wire DynamicReroutingEngine: evaluate cross-track error against active route
    output.cross_track_error_m = rerouting_engine_.ComputeCrossTrackError(
        output.state.latitude_deg,
        output.state.longitude_deg,
        active_route_
    );
    output.is_reroute_needed = rerouting_engine_.IsRerouteNeeded();
    output.is_dislodged = rerouting_engine_.IsDislodged();

    return output;
}

std::optional<SnappedResult> NavicorePipeline::GetSnappedRoad() const {
    return last_snapped_road_;
}

bool NavicorePipeline::IsRerouteNeeded() const {
    return rerouting_engine_.IsRerouteNeeded();
}

bool NavicorePipeline::IsDislodged() const {
    return rerouting_engine_.IsDislodged();
}

} // namespace navicore
