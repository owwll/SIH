#include "navicore/fusion/eskf.hpp"
#include <algorithm>

namespace navicore {

constexpr double EARTH_RADIUS = 6378137.0;
constexpr double DEG_TO_RAD = 3.14159265358979323846 / 180.0;
constexpr double RAD_TO_DEG = 180.0 / 3.14159265358979323846;

EskfFilter::EskfFilter() {
    // Initialize error state covariance diagonals: [p, v, theta, ba, bg]
    p_diag_.fill(1.0f);
    p_diag_[0] = p_diag_[1] = p_diag_[2] = 5.0f;     // Position: 5m
    p_diag_[3] = p_diag_[4] = p_diag_[5] = 0.5f;     // Velocity: 0.5 m/s
    p_diag_[6] = p_diag_[7] = p_diag_[8] = 0.05f;    // Attitude: 0.05 rad
    p_diag_[9] = p_diag_[10] = p_diag_[11] = 0.02f;  // Accel bias
    p_diag_[12] = p_diag_[13] = p_diag_[14] = 0.005f;// Gyro bias
}

void EskfFilter::Predict(const ImuSample& imu, float dt) {
    current_timestamp_nanos_ = imu.timestamp_nanos;

    // 1. Correct sensor inputs with learned biases
    float ax_c = imu.ax - ba_[0];
    float ay_c = imu.ay - ba_[1];
    float az_c = imu.az - ba_[2];
    float gz_c = imu.gz - bg_[2];

    // 2. Track vertical acceleration samples in rolling buffer for road roughness metric
    recent_az_samples_[az_sample_idx_] = az_c;
    az_sample_idx_ = (az_sample_idx_ + 1) % K_ACCEL_WINDOW_SIZE;
    if (az_sample_count_ < K_ACCEL_WINDOW_SIZE) {
        az_sample_count_++;
    }

    // 3. Dynamic acceleration deviation for Spectral ZUPT engine
    // In vehicle frame (Z down), gravity offset is ~9.81 m/s^2.
    // Preserve harmonic oscillations by selecting the dominant dynamic axis
    float dyn_az = az_c - 9.80665f;
    float dyn_accel = (std::abs(dyn_az) >= std::abs(ax_c)) ? dyn_az : ax_c;
    last_spectral_zupt_ = spectral_zupt_engine_.ProcessSample(dyn_accel);

    // If native spectral detector confirms engine idle harmonics, immediately lock
    if (last_spectral_zupt_) {
        ApplyZupt();
    }

    // 4. Propagate attitude (Yaw angle integration)
    yaw_ += gz_c * dt;
    // Normalize yaw to [-pi, pi]
    while (yaw_ > 3.14159265f) yaw_ -= 2.0f * 3.14159265f;
    while (yaw_ < -3.14159265f) yaw_ += 2.0f * 3.14159265f;

    // 5. Propagate velocity in NED frame
    if (current_mode_ == FusionMode::OPEN_SKY) {
        float cos_yaw = std::cos(yaw_);
        float sin_yaw = std::sin(yaw_);
        float an = cos_yaw * ax_c - sin_yaw * ay_c;
        float ae = sin_yaw * ax_c + cos_yaw * ay_c;

        vn_ += an * dt;
        ve_ += ae * dt;
    }

    // 6. Propagate position
    float dn = vn_ * dt;
    float de = ve_ * dt;
    LocalNedToGeo(dn, de, lat_, lon_);

    // 7. Propagate Covariance & grow heading uncertainty during blackout
    if (current_mode_ == FusionMode::DEAD_RECKONING) {
        heading_uncertainty_rad_ += 0.0002f * dt; // Slow drift growth
    }
}

void EskfFilter::UpdateGnss(const GnssFix& fix) {
    last_gnss_timestamp_nanos_ = fix.timestamp_nanos;
    current_mode_ = FusionMode::OPEN_SKY;
    blackout_start_nanos_ = 0;
    heading_uncertainty_rad_ = 0.01f;

    // Kalman gain for GNSS position update
    float k_pos = 0.3f;
    lat_ = (1.0f - k_pos) * lat_ + k_pos * fix.latitude_deg;
    lon_ = (1.0f - k_pos) * lon_ + k_pos * fix.longitude_deg;

    if (fix.speed_mps.has_value() && fix.heading_deg.has_value()) {
        float speed = fix.speed_mps.value();
        float hdg_rad = fix.heading_deg.value() * static_cast<float>(DEG_TO_RAD);

        float k_vel = 0.4f;
        vn_ = (1.0f - k_vel) * vn_ + k_vel * (speed * std::cos(hdg_rad));
        ve_ = (1.0f - k_vel) * ve_ + k_vel * (speed * std::sin(hdg_rad));

        // Online bias learning during high-confidence open sky
        yaw_ = 0.8f * yaw_ + 0.2f * hdg_rad;
    }
}

void EskfFilter::UpdateAiOdometer(const OdometerOutput& ai_out, float /*heading_rad*/) {
    if (blackout_start_nanos_ == 0) {
        blackout_start_nanos_ = current_timestamp_nanos_;
    }

    // 1. Calculate road roughness metric from recent vertical acceleration variance
    road_roughness_factor_ = ComputeVerticalAccelVariance();

    // 2. Compute road-roughness-adaptive NHC measurement noise (sigma_vy, sigma_vz) using vehicle profile stiffness
    NhcKinematics::ComputeAdaptiveCovariance(
        road_roughness_factor_,
        last_sigma_vy_,
        last_sigma_vz_,
        vehicle_profile_.nhc_lateral_stiffness,
        vehicle_profile_.nhc_vertical_stiffness
    );

    // =========================================================================
    // Multi-Sensor ZUPT Safety Fusion Rule (SECURITY.md §7 & PRD §5.4):
    // -------------------------------------------------------------------------
    // A false-positive ZUPT lock while cruising is catastrophic: it clamps filter
    // velocity to zero and freezes navigation position on maps. Conversely,
    // residual dead-reckoning drift during a stop is slow and bounded.
    //
    // Dual detector inputs:
    // (a) Native SpectralZuptEngine (100 Hz): Ingests dynamic acceleration in
    //     Predict() and identifies 20-35 Hz engine idle harmonic vibrations
    //     when translational energy is strictly < 0.15 m/s^2 (last_spectral_zupt_).
    // (b) AI Odometer Classifier (10 Hz): 1D-TCN head estimating stopped_prob.
    //
    // Fusion Policy:
    // - ZUPT engages if EITHER:
    //   1. Native SpectralZuptEngine detects sustained idle harmonics
    //      (last_spectral_zupt_ == true), EVEN IF the AI classifier disagrees
    //      (e.g., stopped_prob is low or 0.0 due to inference latency or model glitch).
    //   2. AI classifier reports high-confidence stop (stopped_prob > 0.85f),
    //      which safeguards Electric Vehicles (EVs) or vehicles with engine
    //      start-stop systems lacking ICE idle harmonics.
    // - Release Policy:
    //   When translational movement resumes (spectral idle harmonics disappear
    //   and AI reports stopped_prob <= 0.85), the filter transitions back to
    //   DEAD_RECKONING, restoring velocity propagation without position freeze.
    // =========================================================================
    bool should_zupt = last_spectral_zupt_ || (ai_out.stopped_prob > 0.85f);

    if (should_zupt) {
        ApplyZupt();
        return;
    }

    current_mode_ = FusionMode::DEAD_RECKONING;

    // Project forward velocity Vx and Non-Holonomic Constraints (Vy=0, Vz=0) to NED frame
    float forward_speed = ai_out.vx_mps;
    float cos_yaw = std::cos(yaw_);
    float sin_yaw = std::sin(yaw_);

    // In dead reckoning, velocity is directly driven by the AI Virtual Odometer
    vn_ = forward_speed * cos_yaw;
    ve_ = forward_speed * sin_yaw;
    vd_ = 0.0f; // NHC: Vertical motion zeroed

    // 3. Update velocity error covariance diagonals in NED frame using AI forward
    // uncertainty and road-roughness-adaptive NHC noise
    float var_vx = ai_out.variance_vx;
    float var_vy = last_sigma_vy_ * last_sigma_vy_;
    float var_vz = last_sigma_vz_ * last_sigma_vz_;

    p_diag_[3] = var_vx * cos_yaw * cos_yaw + var_vy * sin_yaw * sin_yaw;
    p_diag_[4] = var_vx * sin_yaw * sin_yaw + var_vy * cos_yaw * cos_yaw;
    p_diag_[5] = var_vz;
}

void EskfFilter::ApplyZupt() {
    current_mode_ = FusionMode::ZUPT_LOCKED;
    vn_ = 0.0f;
    ve_ = 0.0f;
    vd_ = 0.0f;
    p_diag_[3] = p_diag_[4] = p_diag_[5] = 1e-4f; // Reset velocity error covariance
}

float EskfFilter::ComputeVerticalAccelVariance() const {
    if (az_sample_count_ < 2) {
        return 0.0f;
    }
    float sum = 0.0f;
    for (size_t i = 0; i < az_sample_count_; ++i) {
        sum += recent_az_samples_[i];
    }
    float mean = sum / static_cast<float>(az_sample_count_);

    float var_sum = 0.0f;
    for (size_t i = 0; i < az_sample_count_; ++i) {
        float diff = recent_az_samples_[i] - mean;
        var_sum += diff * diff;
    }
    return var_sum / static_cast<float>(az_sample_count_);
}

FusionState EskfFilter::GetState() const {
    FusionState state;
    state.timestamp_nanos = current_timestamp_nanos_;
    state.latitude_deg = lat_;
    state.longitude_deg = lon_;
    state.altitude_m = alt_;
    state.speed_mps = std::sqrt(vn_ * vn_ + ve_ * ve_);
    state.heading_rad = yaw_;
    state.heading_uncertainty_rad = heading_uncertainty_rad_;
    state.mode = current_mode_;
    
    if (blackout_start_nanos_ > 0 && current_timestamp_nanos_ > blackout_start_nanos_) {
        state.blackout_duration_ms = (current_timestamp_nanos_ - blackout_start_nanos_) / 1000000LL;
    } else {
        state.blackout_duration_ms = 0;
    }

    state.within_validated_range = (state.blackout_duration_ms <= 120000LL); // 120s limit

    return state;
}

void EskfFilter::SetVehicleProfile(const VehicleKinematicProfile& profile) {
    vehicle_profile_ = profile;
    spectral_zupt_engine_.SetIdleFrequencyBand(profile.engine_idle_band_low_hz, profile.engine_idle_band_high_hz);
}

void EskfFilter::LocalNedToGeo(float dn, float de, double& lat, double& lon) const {
    double d_lat = (dn / EARTH_RADIUS) * RAD_TO_DEG;
    double d_lon = (de / (EARTH_RADIUS * std::cos(lat * DEG_TO_RAD))) * RAD_TO_DEG;
    lat += d_lat;
    lon += d_lon;
}

} // namespace navicore
