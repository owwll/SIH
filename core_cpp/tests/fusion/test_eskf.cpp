#include <iostream>
#include <cassert>
#include <cmath>
#include "navicore/types.hpp"
#include "navicore/fusion/eskf.hpp"
#include "navicore/calibration/auto_calib.hpp"
#include "navicore/zupt/nhc_zupt.hpp"
#include "navicore/mapmatch/hmm_matcher.hpp"
#include "navicore/pipeline/navicore_pipeline.hpp"

void TestAutoCalibration() {
    std::cout << "[+] Running TestAutoCalibration..." << std::endl;
    navicore::MountCalibrator calibrator;

    // Simulate phone tilted at 45 degrees pitch
    float gravity_mag = 9.81f;
    float tilt_angle = 3.14159f / 4.0f; // 45 deg

    for (int i = 0; i < 200; ++i) {
        navicore::ImuSample s;
        s.timestamp_nanos = i * 10000000LL;
        // Gravity split between Y and Z
        s.ax = 0.0f;
        s.ay = gravity_mag * std::sin(tilt_angle);
        s.az = gravity_mag * std::cos(tilt_angle);

        // Inject dynamic acceleration along vehicle X
        float forward_accel = (i > 50 && i < 150) ? 1.5f : 0.0f;
        s.ax += forward_accel;

        calibrator.Ingest(s);
    }

    auto rot = calibrator.CurrentRotation();
    assert(rot.has_value() && "Dynamic auto-calibration should resolve rotation matrix R_b^v");
    std::cout << "    -> Calibration Matrix R_b^v resolved successfully!" << std::endl;
}

void TestEskfTunnelBlackout() {
    std::cout << "[+] Running TestEskfTunnelBlackout (60s Blackout Simulation)..." << std::endl;
    navicore::EskfFilter eskf;

    // 1. Warm-up in Open Sky for 10 seconds @ 15 m/s
    float dt = 0.01f; // 100 Hz
    int warmup_steps = 1000;
    for (int i = 0; i < warmup_steps; ++i) {
        navicore::ImuSample s;
        s.timestamp_nanos = i * 10000000LL;
        s.ax = 0.0f; s.ay = 0.0f; s.az = 9.81f;
        s.gx = 0.0f; s.gy = 0.0f; s.gz = 0.0f;
        eskf.Predict(s, dt);

        if (i % 10 == 0) { // 10 Hz GNSS
            navicore::GnssFix fix;
            fix.timestamp_nanos = s.timestamp_nanos;
            fix.latitude_deg = 19.0760 + (i * dt * 15.0) / 111111.0;
            fix.longitude_deg = 72.8777;
            fix.speed_mps = 15.0f;
            fix.heading_deg = 0.0f; // Heading North
            eskf.UpdateGnss(fix);
        }
    }

    navicore::FusionState state_open = eskf.GetState();
    assert(state_open.mode == navicore::FusionMode::OPEN_SKY);

    // 2. Enter 60-Second Tunnel Blackout (No GNSS, AI Odometer at 15 m/s)
    int tunnel_steps = 6000;
    for (int i = 0; i < tunnel_steps; ++i) {
        int64_t t_nano = (warmup_steps + i) * 10000000LL;
        navicore::ImuSample s;
        s.timestamp_nanos = t_nano;
        s.ax = 0.0f; s.ay = 0.0f; s.az = 9.81f;
        s.gx = 0.0f; s.gy = 0.0f; s.gz = 0.0f;
        eskf.Predict(s, dt);

        if (i % 10 == 0) { // 10 Hz AI Odometer
            navicore::OdometerOutput ai;
            ai.vx_mps = 15.0f; // 15 m/s forward speed
            ai.variance_vx = 0.02f;
            ai.stopped_prob = 0.0f;
            eskf.UpdateAiOdometer(ai, 0.0f);
        }
    }

    navicore::FusionState state_blackout = eskf.GetState();
    assert(state_blackout.mode == navicore::FusionMode::DEAD_RECKONING);
    assert(state_blackout.blackout_duration_ms >= 59000);
    assert(state_blackout.within_validated_range == true);

    // Theoretical distance inside tunnel = 15 m/s * 60s = 900 meters
    double d_lat = state_blackout.latitude_deg - state_open.latitude_deg;
    double dist_covered = d_lat * 111111.0;
    double drift_error = std::abs(dist_covered - 900.0);

    std::cout << "    -> Distance traversed in tunnel: " << dist_covered << " m (Expected: 900 m)" << std::endl;
    std::cout << "    -> Total Tunnel Dead Reckoning Error: " << drift_error << " m (< 0.5% drift)" << std::endl;
    assert(drift_error < 5.0 && "Tunnel dead reckoning error must be under 5 meters");
}

void TestZuptTrafficLock() {
    std::cout << "[+] Running TestZuptTrafficLock..." << std::endl;
    navicore::EskfFilter eskf;
    navicore::SpectralZuptEngine zupt_engine;

    // Simulate 22 Hz engine idle vibration during traffic stop
    bool detected_idle = false;
    for (int i = 0; i < 100; ++i) {
        float t = i * 0.01f;
        float idle_vib = 0.6f * std::sin(2.0f * 3.14159f * 22.0f * t);
        if (zupt_engine.ProcessSample(idle_vib)) {
            detected_idle = true;
            eskf.ApplyZupt();
        }
    }

    assert(detected_idle && "Spectral ZUPT must identify 22 Hz engine idle harmonics");
    navicore::FusionState s = eskf.GetState();
    assert(s.mode == navicore::FusionMode::ZUPT_LOCKED);
    assert(s.speed_mps < 1e-4f);
    std::cout << "    -> ZUPT Traffic Stop Lock successfully clamped speed to 0.00 m/s!" << std::endl;
}

void TestSpectralZuptAutonomousTrigger() {
    std::cout << "[+] Running TestSpectralZuptAutonomousTrigger (Spectral ZUPT vs AI disagreement)..." << std::endl;
    navicore::EskfFilter eskf;

    // Simulate 22 Hz engine idle vibration (harmonic ratio > 2.5, translation < 0.15)
    // while AI classifier actively disagrees (stopped_prob = 0.0, vx = 1.0 m/s)
    float dt = 0.01f; // 100 Hz
    for (int i = 0; i < 100; ++i) {
        float t = i * dt;
        float idle_vib = 0.6f * std::sin(2.0f * 3.14159f * 22.0f * t);

        navicore::ImuSample s;
        s.timestamp_nanos = i * 10000000LL;
        s.ax = 0.0f;
        s.ay = 0.0f;
        s.az = 9.80665f + idle_vib; // idle harmonics injected on vertical axis
        s.gx = 0.0f; s.gy = 0.0f; s.gz = 0.0f;

        // Predict feeds dynamic acceleration to native SpectralZuptEngine
        eskf.Predict(s, dt);

        // AI classifier runs at 10 Hz and claims the vehicle is moving (stopped_prob = 0.0)
        if (i % 10 == 0) {
            navicore::OdometerOutput ai;
            ai.vx_mps = 1.0f; // Disagreeing AI says vehicle moving at 1.0 m/s
            ai.variance_vx = 0.05f;
            ai.stopped_prob = 0.0f; // Disagreeing AI reports moving
            eskf.UpdateAiOdometer(ai, 0.0f);
        }
    }

    assert(eskf.IsSpectralZuptActive() && "Spectral engine must detect 22 Hz idle harmonics autonomously");
    navicore::FusionState state = eskf.GetState();
    assert(state.mode == navicore::FusionMode::ZUPT_LOCKED && "Filter must lock to ZUPT even when AI stopped_prob == 0.0");
    assert(state.speed_mps < 1e-4f && "Speed must be clamped to 0.00 m/s by spectral ZUPT");
    std::cout << "    -> Spectral engine successfully triggered ZUPT autonomously while AI classifier disagreed!" << std::endl;
    std::cout << "    -> Fused Speed: " << state.speed_mps << " m/s (Mode: ZUPT_LOCKED)" << std::endl;
}

void TestNhcAdaptiveCovarianceRoughRoad() {
    std::cout << "[+] Running TestNhcAdaptiveCovarianceRoughRoad (Smooth vs Rough Road)..." << std::endl;
    float dt = 0.01f; // 100 Hz

    // 1. Simulate Smooth Road (az tightly clustered around 1g, low variance)
    navicore::EskfFilter eskf_smooth;
    for (int i = 0; i < 50; ++i) {
        navicore::ImuSample s;
        s.timestamp_nanos = i * 10000000LL;
        s.ax = 0.0f; s.ay = 0.0f;
        s.az = 9.80665f + 0.01f * std::sin(i * 0.2f); // Minimal vertical noise
        s.gx = 0.0f; s.gy = 0.0f; s.gz = 0.0f;
        eskf_smooth.Predict(s, dt);
    }
    navicore::OdometerOutput ai_smooth;
    ai_smooth.vx_mps = 15.0f;
    ai_smooth.variance_vx = 0.02f;
    ai_smooth.stopped_prob = 0.0f;
    eskf_smooth.UpdateAiOdometer(ai_smooth, 0.0f);

    float vy_smooth = 0.0f, vz_smooth = 0.0f;
    eskf_smooth.GetLastNhcNoise(vy_smooth, vz_smooth);
    float roughness_smooth = eskf_smooth.GetRoadRoughness();
    auto cov_smooth = eskf_smooth.GetCovarianceDiagonals();
    float p_vz_smooth = cov_smooth[5]; // Var(Vz) = sigma_vz^2

    // 2. Simulate Rough Road (large vertical acceleration fluctuations from potholes / bumps)
    navicore::EskfFilter eskf_rough;
    for (int i = 0; i < 50; ++i) {
        navicore::ImuSample s;
        s.timestamp_nanos = i * 10000000LL;
        s.ax = 0.0f; s.ay = 0.0f;
        // High vertical acceleration variance (+- 2.5 m/s^2 oscillation)
        s.az = 9.80665f + 2.5f * std::sin(2.0f * 3.14159f * 8.0f * (i * dt));
        s.gx = 0.0f; s.gy = 0.0f; s.gz = 0.0f;
        eskf_rough.Predict(s, dt);
    }
    navicore::OdometerOutput ai_rough;
    ai_rough.vx_mps = 15.0f;
    ai_rough.variance_vx = 0.02f;
    ai_rough.stopped_prob = 0.0f;
    eskf_rough.UpdateAiOdometer(ai_rough, 0.0f);

    float vy_rough = 0.0f, vz_rough = 0.0f;
    eskf_rough.GetLastNhcNoise(vy_rough, vz_rough);
    float roughness_rough = eskf_rough.GetRoadRoughness();
    auto cov_rough = eskf_rough.GetCovarianceDiagonals();
    float p_vz_rough = cov_rough[5]; // Var(Vz) = sigma_vz^2

    std::cout << "    [Smooth Road] Vertical Variance: " << roughness_smooth
              << " (m/s^2)^2 | NHC sigma_vy: " << vy_smooth
              << " m/s | NHC sigma_vz: " << vz_smooth
              << " m/s | Cov Var(Vz): " << p_vz_smooth << std::endl;

    std::cout << "    [Rough Road]  Vertical Variance: " << roughness_rough
              << " (m/s^2)^2 | NHC sigma_vy: " << vy_rough
              << " m/s | NHC sigma_vz: " << vz_rough
              << " m/s | Cov Var(Vz): " << p_vz_rough << std::endl;

    // Verify roughness metric detected the road surface difference
    assert(roughness_rough > 10.0f * roughness_smooth && "Rough road vertical variance must exceed smooth road");

    // Verify NHC measurement noise visibly widened
    assert(vz_rough > 3.0f * vz_smooth && "Adaptive NHC sigma_vz must visibly widen under rough road conditions");
    assert(vy_rough > 3.0f * vy_smooth && "Adaptive NHC sigma_vy must visibly widen under rough road conditions");

    // Verify filter velocity error covariance visibly widened (>9x variance)
    assert(p_vz_rough > 9.0f * p_vz_smooth && "Velocity error covariance Var(Vz) must visibly widen (>9x)");

    std::cout << "    -> NHC Adaptive Covariance visibly widened ("
              << (vz_rough / vz_smooth) << "x noise std dev, "
              << (p_vz_rough / p_vz_smooth) << "x variance) under rough-road input!" << std::endl;
}

void TestNavicorePipelineOrchestrator() {
    std::cout << "[+] Running TestNavicorePipelineOrchestrator (TRD §6.1 Full Orchestration)..." << std::endl;

    // 1. Construct pipeline with Sedan vehicle profile
    navicore::NavicorePipeline pipeline(navicore::VehicleType::PASSENGER_SEDAN);
    assert(pipeline.GetVehicleProfile().type == navicore::VehicleType::PASSENGER_SEDAN);
    assert(pipeline.GetVehicleProfile().nhc_lateral_stiffness == 0.98f);

    // 2. Load road network for HMM Map Matching
    navicore::RoadSegment seg{"SEG_BANDRA_01", 19.0760, 72.8777, 19.0780, 72.8777, 0.0f, 222.0f};
    pipeline.SetRoadNetwork({seg});

    // 3. Set active navigation route polyline (Northbound along longitude 72.8777)
    pipeline.SetActiveRoute({
        {19.0760, 72.8777},
        {19.0800, 72.8777}
    });

    // 4. Ingest healthy GNSS fix on road
    navicore::GnssFix fix;
    fix.timestamp_nanos = 1000000000LL;
    fix.latitude_deg = 19.0765;
    fix.longitude_deg = 72.8777;
    fix.speed_mps = 12.0f;
    fix.heading_deg = 0.0f;
    pipeline.UpdateGnss(fix);

    // 5. Ingest IMU sample
    navicore::ImuSample imu;
    imu.timestamp_nanos = 1010000000LL;
    imu.ax = 0.0f; imu.ay = 0.0f; imu.az = 9.80665f;
    imu.gx = 0.0f; imu.gy = 0.0f; imu.gz = 0.0f;
    pipeline.ProcessImu(imu, 0.01f);

    // 6. Test GetState() composite output (ESKF state + HMM snapped road + route tracking)
    navicore::PipelineOutput out = pipeline.GetState();
    assert(out.snapped_road.has_value() && "HMM Map-Matching must snap to SEG_BANDRA_01");
    assert(out.snapped_road->road_id == "SEG_BANDRA_01");
    assert(out.is_reroute_needed == false && "On-route position should not trigger re-route");
    assert(out.is_dislodged == false && "No mount dislodgement on quiescent gyro");
    std::cout << "    -> HMM Map-Matching successfully snapped to: " << out.snapped_road->road_id
              << " (Confidence: " << out.snapped_road->confidence << ")" << std::endl;

    // 7. Test Phone Mount Dislodgement Detection
    navicore::ImuSample jolt = imu;
    jolt.gx = 3.2f; // Sudden 3.2 rad/s roll rate (> 2.5 threshold)
    pipeline.ProcessImu(jolt, 0.01f);
    navicore::PipelineOutput out_jolt = pipeline.GetState();
    assert(out_jolt.is_dislodged == true && "Mount dislodgement must be detected");
    assert(pipeline.IsDislodged() == true);
    std::cout << "    -> Mount dislodgement detected on high-rate gyro shock!" << std::endl;

    // 8. Test Dynamic Re-routing Detection on large cross-track error
    navicore::GnssFix off_route = fix;
    // Displace vehicle 150 meters East off the corridor
    off_route.longitude_deg = 72.8777 + (150.0 / 111319.5);
    pipeline.UpdateGnss(off_route);
    navicore::PipelineOutput out_off = pipeline.GetState();
    assert(out_off.is_reroute_needed == true && "Re-route must be flagged when cross-track error > 30m");
    assert(pipeline.IsRerouteNeeded() == true);
    std::cout << "    -> Re-routing triggered! Cross-Track Error: " << out_off.cross_track_error_m << " m" << std::endl;

    // 9. Test Vehicle Profile Switching (Two-Wheeler motorcycle vs Truck)
    pipeline.SetVehicleType(navicore::VehicleType::TWO_WHEELER_MOTORBIKE);
    assert(pipeline.GetVehicleProfile().type == navicore::VehicleType::TWO_WHEELER_MOTORBIKE);
    assert(pipeline.GetVehicleProfile().nhc_lateral_stiffness == 0.80f);
    assert(pipeline.GetVehicleProfile().engine_idle_band_low_hz == 18.0f);

    pipeline.SetVehicleType(navicore::VehicleType::COMMERCIAL_TRUCK);
    assert(pipeline.GetVehicleProfile().type == navicore::VehicleType::COMMERCIAL_TRUCK);
    assert(pipeline.GetVehicleProfile().nhc_lateral_stiffness == 0.99f);
    assert(pipeline.GetVehicleProfile().engine_idle_band_low_hz == 10.0f);
    std::cout << "    -> Vehicle profile switching (Two-Wheeler vs Truck) verified!" << std::endl;
}

int main() {
    std::cout << "====================================================" << std::endl;
    std::cout << "[*] NAVICORE AI: NATIVE C++ MATH ENGINE TEST SUITE  " << std::endl;
    std::cout << "====================================================" << std::endl;

    TestAutoCalibration();
    TestEskfTunnelBlackout();
    TestZuptTrafficLock();
    TestSpectralZuptAutonomousTrigger();
    TestNhcAdaptiveCovarianceRoughRoad();
    TestNavicorePipelineOrchestrator();

    std::cout << "====================================================" << std::endl;
    std::cout << "[*] ALL NATIVE C++ INTEGRATION TESTS PASSED (100%)! " << std::endl;
    std::cout << "====================================================" << std::endl;
    return 0;
}
