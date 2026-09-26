#include <jni.h>
#include <memory>
#include "navicore/fusion/eskf.hpp"
#include "navicore/calibration/auto_calib.hpp"

static std::unique_ptr<navicore::EskfFilter> g_eskf = nullptr;
static std::unique_ptr<navicore::MountCalibrator> g_calibrator = nullptr;

extern "C" {

JNIEXPORT void JNICALL
Java_org_enigma_navicore_fusion_FusionCore_nativeInit(JNIEnv* /*env*/, jobject /*thiz*/) {
    g_eskf = std::make_unique<navicore::EskfFilter>();
    g_calibrator = std::make_unique<navicore::MountCalibrator>();
}

JNIEXPORT void JNICALL
Java_org_enigma_navicore_fusion_FusionCore_nativeProcessImu(
    JNIEnv* /*env*/,
    jobject /*thiz*/,
    jlong timestampNanos,
    jfloat ax, jfloat ay, jfloat az,
    jfloat gx, jfloat gy, jfloat gz,
    jfloat dt
) {
    if (!g_eskf || !g_calibrator) return;

    navicore::ImuSample sample;
    sample.timestamp_nanos = timestampNanos;
    sample.ax = ax; sample.ay = ay; sample.az = az;
    sample.gx = gx; sample.gy = gy; sample.gz = gz;

    g_calibrator->Ingest(sample);

    float vx, vy, vz;
    g_calibrator->TransformBodyToVehicle(ax, ay, az, vx, vy, vz);
    sample.ax = vx; sample.ay = vy; sample.az = vz;

    g_eskf->Predict(sample, dt);
}

JNIEXPORT void JNICALL
Java_org_enigma_navicore_fusion_FusionCore_nativeUpdateGnss(
    JNIEnv* /*env*/,
    jobject /*thiz*/,
    jlong timestampNanos,
    jdouble lat, jdouble lon, jdouble alt,
    jfloat speedMps, jfloat headingDeg
) {
    if (!g_eskf) return;

    navicore::GnssFix fix;
    fix.timestamp_nanos = timestampNanos;
    fix.latitude_deg = lat;
    fix.longitude_deg = lon;
    fix.altitude_m = alt;
    fix.speed_mps = speedMps;
    fix.heading_deg = headingDeg;

    g_eskf->UpdateGnss(fix);
}

JNIEXPORT void JNICALL
Java_org_enigma_navicore_fusion_FusionCore_nativeUpdateAiOdometer(
    JNIEnv* /*env*/,
    jobject /*thiz*/,
    jfloat vxMps,
    jfloat varianceVx,
    jfloat stoppedProb
) {
    if (!g_eskf) return;

    if (stoppedProb > 0.85f) {
        g_eskf->ApplyZupt();
    } else {
        navicore::OdometerOutput out;
        out.vx_mps = vxMps;
        out.variance_vx = varianceVx;
        out.stopped_prob = stoppedProb;
        g_eskf->UpdateAiOdometer(out, 0.0f);
    }
}

JNIEXPORT jdoubleArray JNICALL
Java_org_enigma_navicore_fusion_FusionCore_nativeGetState(JNIEnv* env, jobject /*thiz*/) {
    if (!g_eskf) return nullptr;

    navicore::FusionState s = g_eskf->GetState();

    jdoubleArray result = env->NewDoubleArray(8);
    jdouble buffer[8];
    buffer[0] = s.latitude_deg;
    buffer[1] = s.longitude_deg;
    buffer[2] = s.altitude_m;
    buffer[3] = static_cast<jdouble>(s.speed_mps);
    buffer[4] = static_cast<jdouble>(s.heading_rad);
    buffer[5] = static_cast<jdouble>(s.heading_uncertainty_rad);
    buffer[6] = static_cast<jdouble>(static_cast<int>(s.mode));
    buffer[7] = static_cast<jdouble>(s.blackout_duration_ms);

    env->SetDoubleArrayRegion(result, 0, 8, buffer);
    return result;
}

} // extern "C"
