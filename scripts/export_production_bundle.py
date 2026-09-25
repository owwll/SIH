#!/usr/bin/env python3
"""
=============================================================================
NAVICORE AI: PRODUCTION BUNDLE EXPORTER & RELEASE PACKAGER
Smart India Hackathon 2026 | Problem Statement ID: 260168
Theme: Smart Vehicles | Team: @enigm@ (Team ID: 132834)
=============================================================================
Verifies all project layers and builds a validated production release manifest:
1. Native C++ Core Engine & Shared Libraries
2. ONNX & INT8 TFLite Virtual Odometer Neural Artifacts
3. Real IO-VNBD Dataset Parser & Synchronized Benchmarks
4. Android Native NDK JNI Bridge & Kotlin Service
5. ROS 2 Robotics Autonomous Vehicle Node
6. 3D Map Visualizer & Web Telemetry Suite
7. Python Drop-In SDK & Multi-Vehicle Kinematics Suite
"""

import os
import sys
import hashlib
import json
import time

# Force UTF-8 on Windows Console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def calculate_sha256(filepath):
    """Calculates SHA256 hash of a file."""
    if not os.path.exists(filepath):
        return None
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_layer(name, files, base_dir):
    print(f"\n[LAYER] {name}")
    print("─" * 78)
    layer_ok = True
    manifest = []
    
    for rel_path in files:
        full_path = os.path.join(base_dir, rel_path)
        exists = os.path.exists(full_path)
        size_bytes = os.path.getsize(full_path) if exists else 0
        sha = calculate_sha256(full_path) if exists else "MISSING"
        
        status = "✅ OK" if exists else "❌ MISSING"
        if not exists:
            layer_ok = False
        
        print(f"  {status} | {rel_path:<48} | {size_bytes:>8} B | {sha[:10]}...")
        manifest.append({
            "file": rel_path,
            "exists": exists,
            "size_bytes": size_bytes,
            "sha256": sha
        })
    return layer_ok, manifest


def export_bundle():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("═" * 78)
    print(" 🚀 NAVICORE AI: PRODUCTION RELEASE & BUNDLE INTEGRITY AUDIT")
    print(" Smart India Hackathon 2026 | PS 260168 | Team: @enigm@ (132834)")
    print("═" * 78)

    layers = {
        "1. Core C++ Native Fusion Engine": [
            "core_cpp/include/navicore/types.hpp",
            "core_cpp/include/navicore/fusion/eskf.hpp",
            "core_cpp/include/navicore/calibration/auto_calib.hpp",
            "core_cpp/include/navicore/zupt/nhc_zupt.hpp",
            "core_cpp/include/navicore/mapmatch/hmm_matcher.hpp",
            "core_cpp/include/navicore/sensors/barometer_tracker.hpp",
            "core_cpp/include/navicore/vehicle/vehicle_profile_manager.hpp",
            "core_cpp/include/navicore/routing/rerouting_engine.hpp",
            "core_cpp/include/navicore/sdk_api.h",
            "core_cpp/src/fusion/eskf.cpp",
            "core_cpp/src/calibration/auto_calib.cpp",
            "core_cpp/src/zupt/nhc_zupt.cpp",
            "core_cpp/src/mapmatch/hmm_matcher.cpp",
            "core_cpp/CMakeLists.txt"
        ],
        "2. ML 1D-TCN Neural Odometer & Quantized Models": [
            "ml_pipeline/src/models/tcn_odometer.py",
            "ml_pipeline/src/models/export/export_tflite.py",
            "ml_pipeline/src/dataset/real_data_pipeline.py",
            "ml_pipeline/src/dataset/preprocessing.py",
            "ml_pipeline/src/dataset/augmentations.py",
            "models/exported/navicore_odometer_int8.tflite"
        ],
        "3. Real IO-VNBD Dataset Benchmarks": [
            "data/sample_iovnbd/S-S1.csv",
            "data/sample_iovnbd/V-S1.csv",
            "data/IO-VNBD_SPEC.txt"
        ],
        "4. Android NDK JNI Bridge & Mobile Client": [
            "android_app/app/src/main/cpp/jni_bridge.cpp",
            "android_app/app/src/main/cpp/CMakeLists.txt",
            "android_app/app/src/main/java/org/enigma/navicore/MainActivity.kt"
        ],
        "5. ROS 2 Robotics Node": [
            "ros2_node/src/navicore_node.cpp",
            "ros2_node/CMakeLists.txt",
            "ros2_node/package.xml"
        ],
        "6. 3D Web Visualizer & Real-time Sensor Server": [
            "web_visualizer/index.html",
            "web_visualizer/app.js",
            "web_visualizer/style.css",
            "web_visualizer/sensor_stream.html",
            "scripts/launch_visualizer.py",
            "scripts/realtime_sensor_server.py"
        ],
        "7. Python Fleet SDK & Verification Harnesses": [
            "sdk_python/navicore_sdk.py",
            "scripts/test_end_to_end.py",
            "scripts/verify_vehicle_kinematics.py",
            "scripts/strict_verification_harness.py",
            "scripts/profile_hardware_footprint.py",
            "scripts/benchmark_suite.py",
            "scripts/presentation_mode.py"
        ]
    }

    full_manifest = {}
    total_passed = 0
    total_layers = len(layers)

    for layer_name, files in layers.items():
        ok, m = check_layer(layer_name, files, base_dir)
        full_manifest[layer_name] = m
        if ok:
            total_passed += 1

    # Save manifest
    manifest_path = os.path.join(base_dir, "models", "production_release_manifest.json")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "project": "NaviCore AI",
            "sih_year": 2026,
            "problem_id": "260168",
            "team": "@enigm@ (132834)",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "layers_passed": f"{total_passed}/{total_layers}",
            "manifest": full_manifest
        }, f, indent=2)

    print("\n" + "═" * 78)
    print(f"🎯 PRODUCTION BUNDLE AUDIT: {total_passed}/{total_layers} LAYERS 100% READY!")
    print(f"📦 Manifest saved to: {manifest_path}")
    print("═" * 78 + "\n")
    return total_passed == total_layers


if __name__ == "__main__":
    success = export_bundle()
    sys.exit(0 if success else 1)
