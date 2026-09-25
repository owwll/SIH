#include "navicore/calibration/auto_calib.hpp"
#include <algorithm>
#include <numeric>

namespace navicore {

MountCalibrator::MountCalibrator(float lpf_cutoff_hz, float sample_rate_hz) {
    float dt = 1.0f / sample_rate_hz;
    float rc = 1.0f / (2.0f * 3.14159265f * lpf_cutoff_hz);
    alpha_ = dt / (rc + dt);

    // Initialize identity matrix
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            r_b_v_.r[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

void MountCalibrator::Ingest(const ImuSample& sample) {
    // 1. Low-Pass Filter Gravity Extraction
    if (!gravity_initialized_) {
        gravity_b_[0] = sample.ax;
        gravity_b_[1] = sample.ay;
        gravity_b_[2] = sample.az;
        gravity_initialized_ = true;
    } else {
        gravity_b_[0] = alpha_ * sample.ax + (1.0f - alpha_) * gravity_b_[0];
        gravity_b_[1] = alpha_ * sample.ay + (1.0f - alpha_) * gravity_b_[1];
        gravity_b_[2] = alpha_ * sample.az + (1.0f - alpha_) * gravity_b_[2];
    }

    // 2. Dynamic Acceleration Extraction
    float dyn_ax = sample.ax - gravity_b_[0];
    float dyn_ay = sample.ay - gravity_b_[1];
    float dyn_az = sample.az - gravity_b_[2];

    // Compute magnitude of dynamic acceleration
    float dyn_mag = std::sqrt(dyn_ax * dyn_ax + dyn_ay * dyn_ay + dyn_az * dyn_az);

    // Only collect PCA samples during significant vehicle acceleration/braking (> 0.4 m/s^2)
    if (dyn_mag > 0.4f && dyn_mag < 4.0f) {
        horiz_accel_window_.push_back({dyn_ax, dyn_ay});
        if (horiz_accel_window_.size() >= required_pca_samples_) {
            ComputePcaHeading();
            horiz_accel_window_.clear();
        }
    }
}

void MountCalibrator::ComputePcaHeading() {
    if (horiz_accel_window_.empty()) return;

    size_t n = horiz_accel_window_.size();
    float sum_x = 0.0f, sum_y = 0.0f;
    for (const auto& pt : horiz_accel_window_) {
        sum_x += pt.first;
        sum_y += pt.second;
    }
    float mean_x = sum_x / static_cast<float>(n);
    float mean_y = sum_y / static_cast<float>(n);

    // Compute 2x2 Covariance Matrix
    float cxx = 0.0f, cxy = 0.0f, cyy = 0.0f;
    for (const auto& pt : horiz_accel_window_) {
        float dx = pt.first - mean_x;
        float dy = pt.second - mean_y;
        cxx += dx * dx;
        cxy += dx * dy;
        cyy += dy * dy;
    }
    cxx /= static_cast<float>(n);
    cxy /= static_cast<float>(n);
    cyy /= static_cast<float>(n);

    // Principal Eigenvector calculation (Closed form 2x2 symmetric eigenvalue problem)
    float trace = cxx + cyy;
    float det = cxx * cyy - cxy * cxy;
    float lambda_max = trace / 2.0f + std::sqrt(std::max(0.0f, (trace * trace / 4.0f) - det));

    float vx = 1.0f, vy = 0.0f;
    if (std::abs(cxy) > 1e-6f) {
        vx = lambda_max - cyy;
        vy = cxy;
    } else if (cxx < cyy) {
        vx = 0.0f;
        vy = 1.0f;
    }
    float norm = std::sqrt(vx * vx + vy * vy);
    if (norm > 1e-6f) {
        vx /= norm;
        vy /= norm;
    }

    // Gravity vector in body frame forms the vertical Z axis
    float g_norm = std::sqrt(gravity_b_[0]*gravity_b_[0] + gravity_b_[1]*gravity_b_[1] + gravity_b_[2]*gravity_b_[2]);
    if (g_norm < 1e-4f) return;

    float uz[3] = { gravity_b_[0] / g_norm, gravity_b_[1] / g_norm, gravity_b_[2] / g_norm };
    float ux_raw[3] = { vx, vy, 0.0f };

    ConstructOrthonormalBasis(uz, ux_raw);
    is_calibrated_ = true;
    is_recalibrating_ = false;
}

void MountCalibrator::ConstructOrthonormalBasis(const float uz[3], const float ux_raw[3]) {
    // Gram-Schmidt Orthogonalization:
    // uy = uz x ux_raw
    float uy[3] = {
        uz[1] * ux_raw[2] - uz[2] * ux_raw[1],
        uz[2] * ux_raw[0] - uz[0] * ux_raw[2],
        uz[0] * ux_raw[1] - uz[1] * ux_raw[0]
    };
    float uy_norm = std::sqrt(uy[0]*uy[0] + uy[1]*uy[1] + uy[2]*uy[2]);
    if (uy_norm < 1e-6f) return;
    uy[0] /= uy_norm; uy[1] /= uy_norm; uy[2] /= uy_norm;

    // ux = uy x uz
    float ux[3] = {
        uy[1] * uz[2] - uy[2] * uz[1],
        uy[2] * uz[0] - uy[0] * uz[2],
        uy[0] * uz[1] - uy[1] * uz[0]
    };

    // Rotation Matrix R_b^v rows are [ux^T; uy^T; uz^T]
    r_b_v_.r[0][0] = ux[0]; r_b_v_.r[0][1] = ux[1]; r_b_v_.r[0][2] = ux[2];
    r_b_v_.r[1][0] = uy[0]; r_b_v_.r[1][1] = uy[1]; r_b_v_.r[1][2] = uy[2];
    r_b_v_.r[2][0] = uz[0]; r_b_v_.r[2][1] = uz[1]; r_b_v_.r[2][2] = uz[2];
}

std::optional<RotationMatrix3x3> MountCalibrator::CurrentRotation() const {
    if (!is_calibrated_) return std::nullopt;
    return r_b_v_;
}

bool MountCalibrator::IsRecalibrating() const {
    return is_recalibrating_;
}

void MountCalibrator::TransformBodyToVehicle(float bx, float by, float bz, float& vx, float& vy, float& vz) const {
    vx = r_b_v_.r[0][0] * bx + r_b_v_.r[0][1] * by + r_b_v_.r[0][2] * bz;
    vy = r_b_v_.r[1][0] * bx + r_b_v_.r[1][1] * by + r_b_v_.r[1][2] * bz;
    vz = r_b_v_.r[2][0] * bx + r_b_v_.r[2][1] * by + r_b_v_.r[2][2] * bz;
}

} // namespace navicore
