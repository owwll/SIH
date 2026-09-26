#pragma once

#include "navicore/types.hpp"
#include "navicore/zupt/nhc_zupt.hpp"
#include "navicore/vehicle/vehicle_profile_manager.hpp"
#include <cmath>
#include <array>

namespace navicore {

/**
 * @brief 15-State Error-State Kalman Filter (ESKF) for GNSS/INS/AI fusion.
 * Error state: [delta_p (3), delta_v (3), delta_theta (3), delta_ba (3), delta_bg (3)]
 */
class EskfFilter {
public:
    EskfFilter();

    /**
     * @brief High-frequency IMU propagation step (100 Hz).
     */
    void Predict(const ImuSample& imu_vehicle_frame, float dt_seconds);

    /**
     * @brief Measurement update when GNSS is healthy (Open Sky).
     */
    void UpdateGnss(const GnssFix& gnss_fix);

    /**
     * @brief Measurement update when GNSS is in blackout (<10ms hot switch).
     * Fuses AI predicted forward velocity Vx + Non-Holonomic Constraints (Vy=0, Vz=0).
     */
    void UpdateAiOdometer(const OdometerOutput& ai_odometry, float heading_rad);

    /**
     * @brief Zero-Velocity Update (ZUPT) triggered by AI idle classifier or spectral engine.
     */
    void ApplyZupt();

    /**
     * @brief Returns current fused state at 10 Hz.
     */
    FusionState GetState() const;

    /**
     * @brief Configures vehicle kinematic profile (NHC stiffness & engine idle frequency band).
     */
    void SetVehicleProfile(const VehicleKinematicProfile& profile);

    /**
     * @brief Returns current vehicle kinematic profile.
     */
    const VehicleKinematicProfile& GetVehicleProfile() const {
        return vehicle_profile_;
    }

    /**
     * @brief Returns the last computed NHC measurement noise standard deviations.
     */
    void GetLastNhcNoise(float& out_sigma_vy, float& out_sigma_vz) const {
        out_sigma_vy = last_sigma_vy_;
        out_sigma_vz = last_sigma_vz_;
    }

    /**
     * @brief Returns whether the native SpectralZuptEngine detected engine idle harmonics.
     */
    bool IsSpectralZuptActive() const {
        return last_spectral_zupt_;
    }

    /**
     * @brief Returns the computed vertical acceleration variance (road roughness metric).
     */
    float GetRoadRoughness() const {
        return road_roughness_factor_;
    }

    /**
     * @brief Accessor for error state covariance diagonals [15].
     */
    const std::array<float, 15>& GetCovarianceDiagonals() const {
        return p_diag_;
    }

private:
    // Nominal state
    double lat_{19.0760};   // Degrees
    double lon_{72.8777};
    double alt_{10.0};      // Meters
    float vn_{0.0f}, ve_{0.0f}, vd_{0.0f}; // Velocity in NED frame (m/s)
    float roll_{0.0f}, pitch_{0.0f}, yaw_{0.0f}; // Euler angles (radians)
    float ba_[3]{0.0f, 0.0f, 0.0f}; // Accelerometer bias (m/s^2)
    float bg_[3]{0.0f, 0.0f, 0.0f}; // Gyroscope bias (rad/s)

    // Covariance matrix diagonals (simplified 15x15 diagonal tracking for embedded efficiency)
    std::array<float, 15> p_diag_;

    FusionMode current_mode_{FusionMode::OPEN_SKY};
    int64_t last_gnss_timestamp_nanos_{0};
    int64_t blackout_start_nanos_{0};
    int64_t current_timestamp_nanos_{0};

    float heading_uncertainty_rad_{0.01f};

    // Frequency-domain idle-harmonic detector (20-35 Hz)
    SpectralZuptEngine spectral_zupt_engine_{100.0f, 64};
    bool last_spectral_zupt_{false};

    // Rolling window for vertical-acceleration variance (road roughness metric)
    static constexpr size_t K_ACCEL_WINDOW_SIZE = 50;
    std::array<float, K_ACCEL_WINDOW_SIZE> recent_az_samples_{};
    size_t az_sample_count_{0};
    size_t az_sample_idx_{0};
    float road_roughness_factor_{0.0f};

    // Adaptive NHC measurement noise
    float last_sigma_vy_{0.05f};
    float last_sigma_vz_{0.05f};

    // Active vehicle kinematic profile
    VehicleKinematicProfile vehicle_profile_{VehicleProfileManager::GetProfile(VehicleType::PASSENGER_SEDAN)};

    float ComputeVerticalAccelVariance() const;
    void LocalNedToGeo(float dn, float de, double& lat, double& lon) const;
};

} // namespace navicore
