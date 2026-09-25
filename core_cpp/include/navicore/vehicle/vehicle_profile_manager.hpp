#pragma once

#include <string>
#include <unordered_map>

namespace navicore {

enum class VehicleType {
    TWO_WHEELER_MOTORBIKE,
    PASSENGER_SEDAN,
    COMMERCIAL_TRUCK,
    HEAVY_BUS,
    AUTONOMOUS_DELIVERY_ROBOT
};

struct VehicleKinematicProfile {
    VehicleType type;
    std::string name;
    float max_forward_speed_mps;
    float max_acceleration_mps2;
    float max_deceleration_mps2;
    float max_yaw_rate_rads;
    float nhc_lateral_stiffness;    // Weight for lateral velocity penalty (Vy ≈ 0)
    float nhc_vertical_stiffness;   // Weight for vertical velocity penalty (Vz ≈ 0)
    float engine_idle_band_low_hz;  // Spectral ZUPT lower frequency bound
    float engine_idle_band_high_hz; // Spectral ZUPT upper frequency bound
    float banking_roll_tolerance;   // Two-wheeler roll tolerance before NHC relaxation
};

class VehicleProfileManager {
public:
    static VehicleKinematicProfile GetProfile(VehicleType type) {
        switch (type) {
            case VehicleType::TWO_WHEELER_MOTORBIKE:
                return {
                    VehicleType::TWO_WHEELER_MOTORBIKE,
                    "Two-Wheeler (Motorcycle/Scooter)",
                    45.0f,  // ~160 km/h
                    4.5f,   // High acceleration
                    8.0f,   // Sharp braking
                    2.8f,   // High yaw agility
                    0.80f,  // Flexible lateral (permits banking)
                    0.90f,
                    18.0f,  // Single/twin cylinder idle (18-28 Hz)
                    28.0f,
                    0.60f   // High roll tolerance for cornering lean
                };

            case VehicleType::PASSENGER_SEDAN:
                return {
                    VehicleType::PASSENGER_SEDAN,
                    "Passenger Car (Hatchback/Sedan)",
                    55.0f,  // ~200 km/h
                    3.5f,
                    7.0f,
                    1.5f,
                    0.98f,  // Strict lateral constraint (no tire slide)
                    0.98f,
                    15.0f,  // 4-cylinder engine idle (15-25 Hz)
                    25.0f,
                    0.15f   // Low roll tolerance
                };

            case VehicleType::COMMERCIAL_TRUCK:
                return {
                    VehicleType::COMMERCIAL_TRUCK,
                    "Commercial Freight Truck (Heavy)",
                    30.0f,  // ~110 km/h
                    1.8f,
                    4.5f,
                    0.8f,
                    0.99f,  // Rigid lateral constraint
                    0.99f,
                    10.0f,  // Low-speed diesel idle (10-18 Hz)
                    18.0f,
                    0.08f
                };

            case VehicleType::HEAVY_BUS:
                return {
                    VehicleType::HEAVY_BUS,
                    "Urban Transit Bus",
                    25.0f,  // ~90 km/h
                    1.5f,
                    4.0f,
                    0.7f,
                    0.99f,
                    0.99f,
                    10.0f,
                    16.0f,
                    0.06f
                };

            case VehicleType::AUTONOMOUS_DELIVERY_ROBOT:
                return {
                    VehicleType::AUTONOMOUS_DELIVERY_ROBOT,
                    "Autonomous Delivery AGV (Last-Mile)",
                    8.0f,   // ~30 km/h
                    2.0f,
                    5.0f,
                    3.5f,   // Zero-turn radius capability
                    0.95f,
                    0.95f,
                    0.0f,   // Electric motor (no combustion idle peak)
                    0.0f,
                    0.05f
                };
        }
        // Default fallback to sedan
        return GetProfile(VehicleType::PASSENGER_SEDAN);
    }
};

} // namespace navicore
