#pragma once

#include "navicore/types.hpp"
#include <vector>
#include <cmath>

namespace navicore {

/**
 * @brief Zero-touch automated dynamic mounting calibration.
 * Resolves the 3x3 rotation matrix R_b^v from Phone Body frame to Vehicle Frame.
 */
class MountCalibrator {
public:
    explicit MountCalibrator(float lpf_cutoff_hz = 0.5f, float sample_rate_hz = 100.0f);

    /**
     * @brief Ingest a single 6-DOF IMU sample.
     */
    void Ingest(const ImuSample& sample);

    /**
     * @brief Returns current estimated 3x3 rotation matrix R_b^v.
     * @return nullopt if insufficient motion data collected to resolve heading.
     */
    std::optional<RotationMatrix3x3> CurrentRotation() const;

    /**
     * @brief True if a sudden physical dislodgement / re-orientation is detected.
     */
    bool IsRecalibrating() const;

    /**
     * @brief Transforms 3D vector from Phone Body frame {b} to Vehicle frame {v}.
     */
    void TransformBodyToVehicle(float bx, float by, float bz, float& vx, float& vy, float& vz) const;

private:
    float alpha_{0.03f}; // LPF coefficient
    float gravity_b_[3]{0.0f, 0.0f, 9.81f};
    bool gravity_initialized_{false};

    // Rolling window of horizontal dynamic accelerations for PCA
    std::vector<std::pair<float, float>> horiz_accel_window_;
    size_t required_pca_samples_{64}; // 64 samples (0.64s @ 100 Hz)

    RotationMatrix3x3 r_b_v_{};
    bool is_calibrated_{false};
    bool is_recalibrating_{false};

    void ComputePcaHeading();
    void ConstructOrthonormalBasis(const float uz[3], const float ux_raw[3]);
};

} // namespace navicore
