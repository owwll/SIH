#include "navicore/zupt/nhc_zupt.hpp"
#include <numeric>
#include <algorithm>

namespace navicore {

SpectralZuptEngine::SpectralZuptEngine(float sample_rate_hz, size_t window_size)
    : sample_rate_(sample_rate_hz), window_size_(window_size) {
    buffer_.reserve(window_size_);
}

bool SpectralZuptEngine::ProcessSample(float dynamic_accel_mag) {
    buffer_.push_back(dynamic_accel_mag);
    if (buffer_.size() > window_size_) {
        buffer_.erase(buffer_.begin());
    }

    if (buffer_.size() < window_size_) {
        return false;
    }

    // 1. Calculate energy in translation band (0.1 Hz - 5 Hz)
    float e_trans = ComputeBandEnergy(0.1f, 5.0f);

    // 2. Calculate energy in engine idle harmonic band (configurable per vehicle profile)
    float e_idle = ComputeBandEnergy(idle_band_low_hz_, idle_band_high_hz_);

    if (e_trans < 1e-5f) e_trans = 1e-5f;
    last_energy_ratio_ = e_idle / e_trans;

    // If idle harmonics dominate low-frequency translational energy, vehicle is stationary in traffic
    return (last_energy_ratio_ > 2.5f && e_trans < 0.15f);
}

float SpectralZuptEngine::GetHarmonicEnergyRatio() const {
    return last_energy_ratio_;
}

float SpectralZuptEngine::ComputeBandEnergy(float low_hz, float high_hz) const {
    // Discrete Fourier Transform energy accumulation over specified band
    float energy = 0.0f;
    size_t n = buffer_.size();
    float f_res = sample_rate_ / static_cast<float>(n);

    int k_min = static_cast<int>(low_hz / f_res);
    int k_max = static_cast<int>(high_hz / f_res);
    k_min = std::max(0, k_min);
    k_max = std::min(static_cast<int>(n / 2), k_max);

    for (int k = k_min; k <= k_max; ++k) {
        float re = 0.0f;
        float im = 0.0f;
        float omega = 2.0f * 3.14159265f * static_cast<float>(k) / static_cast<float>(n);

        for (size_t t = 0; t < n; ++t) {
            re += buffer_[t] * std::cos(omega * static_cast<float>(t));
            im -= buffer_[t] * std::sin(omega * static_cast<float>(t));
        }
        energy += (re * re + im * im) / static_cast<float>(n * n);
    }

    return energy;
}

} // namespace navicore
