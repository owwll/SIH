#pragma once

#include <vector>
#include <cmath>
#include <deque>

namespace navicore {

/**
 * Barometric Altimetry & Multi-Floor Elevation Tracker.
 * Fuses barometric atmospheric pressure (hPa) with vertical acceleration (Az)
 * to estimate vehicle altitude, grade/slope angle, and multi-floor level changes in parking basements.
 */
class BarometerElevationTracker {
public:
    BarometerElevationTracker(float p0_hpa = 1013.25f, float lpf_alpha = 0.1f)
        : p0_hpa_(p0_hpa), alpha_(lpf_alpha), altitude_m_(0.0f), filtered_pressure_(p0_hpa), floor_level_(0) {}

    /**
     * Converts atmospheric pressure to barometric altitude using the International Standard Atmosphere (ISA) barometric formula.
     * h = 44330 * (1 - (P / P0)^(1 / 5.255))
     */
    float PressureToAltitude(float pressure_hpa) {
        if (pressure_hpa <= 0.0f) return altitude_m_;
        return 44330.0f * (1.0f - std::pow(pressure_hpa / p0_hpa_, 0.190284f));
    }

    /**
     * Updates the elevation tracker with a new barometric pressure reading.
     * @param pressure_hpa Current pressure reading from phone barometer in hPa / mbar.
     * @param vertical_accel_mps2 Decoupled vertical acceleration (Az).
     * @param dt Sampling time delta in seconds.
     * @return Filtered altitude in meters above reference.
     */
    float Update(float pressure_hpa, float vertical_accel_mps2, float dt) {
        if (filtered_pressure_ <= 0.0f) filtered_pressure_ = pressure_hpa;
        filtered_pressure_ = alpha_ * pressure_hpa + (1.0f - alpha_) * filtered_pressure_;

        float raw_altitude = PressureToAltitude(filtered_pressure_);

        // Complementary filter fusion with vertical acceleration
        altitude_m_ = 0.95f * (altitude_m_ + vertical_accel_mps2 * dt * dt * 0.5f) + 0.05f * raw_altitude;

        // Multi-level basement / parking floor detection (assuming standard 3.0m floor height)
        floor_level_ = static_cast<int>(std::floor((altitude_m_ + 1.5f) / 3.0f));

        return altitude_m_;
    }

    float GetAltitude() const { return altitude_m_; }
    int GetFloorLevel() const { return floor_level_; }
    void SetReferencePressure(float p0_hpa) { p0_hpa_ = p0_hpa; }

private:
    float p0_hpa_;
    float alpha_;
    float altitude_m_;
    float filtered_pressure_;
    int floor_level_;
};

} // namespace navicore
