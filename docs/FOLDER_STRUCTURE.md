# Repository Architecture & Directory Blueprint

## Project: NaviCore AI
**Subtitle**: Enterprise-Grade Folder Structure & Modular Codebase Blueprint  
**Smart India Hackathon 2026** | **Problem Statement ID**: 260168  
**Team**: @enigm@ (Team ID: 132834)

---

## 1. Directory Tree Overview

```
c:\Users\Manthan\Desktop\SIH168/
├── .github/
│   └── workflows/
│       ├── build_and_test.yml         # CI/CD: Automated Kotlin, C++, and Python tests
│       └── lint_check.yml             # Automated ktlint, ruff, black, clang-format
│
├── docs/                              # Engineering, Product & Architecture Documentation
│   ├── PRD.md                         # Product Requirements Document (Extra-Detailed)
│   ├── TRD.md                         # Technical Requirements Document (Math & State Equations)
│   ├── AI_ARCHITECTURE.md             # 1D-TCN Neural Odometer & ZUPT Specification
│   ├── ROADMAP.md                     # 6-Phase / 10-Week Execution Plan & WBS
│   ├── API.MD                         # Internal Module Interface Contracts & Signatures
│   ├── CODE-STYLE.md                  # Multi-Language Code Style, Linters & Guidelines
│   ├── SECURITY.md                    # Data Classification, Threat Model & Least Privilege
│   ├── TECH_STACK_AND_TOOLS_GUIDE.md  # Free & Open-Source Tools, Maps & SDK Guide
│   └── FOLDER_STRUCTURE.md            # This document: Architecture & Blueprint
│
├── android_app/                       # Production Android Navigation Application (Kotlin + NDK)
│   ├── app/
│   │   ├── build.gradle.kts           # App-level build configuration (MapLibre, Compose, TFLite)
│   │   ├── proguard-rules.pro         # Proguard rules for TFLite and Native JNI
│   │   └── src/
│   │       ├── main/
│   │       │   ├── AndroidManifest.xml # Permissions (Sensors, Fine Location, Foreground Service)
│   │       │   ├── cpp/               # Android NDK CMake & JNI Bridge
│   │       │   │   ├── CMakeLists.txt # NDK Build script linking C++ Core & Eigen
│   │       │   │   └── jni_bridge.cpp # DirectByteBuffer zero-copy sensor bridge
│   │       │   ├── java/org/enigma/navicore/
│   │       │   │   ├── calibration/   # Phone-to-vehicle dynamic mount calibration (LPF + PCA)
│   │       │   │   │   ├── MountCalibrator.kt
│   │       │   │   │   └── PcaEstimator.kt
│   │       │   │   ├── odometer/      # On-device TFLite neural odometer inference
│   │       │   │   │   ├── VirtualOdometer.kt
│   │       │   │   │   └── TfliteRunner.kt
│   │       │   │   ├── fusion/        # 15-State ESKF, NHC Kinematics & ZUPT State Machine
│   │       │   │   │   ├── FusionCore.kt
│   │       │   │   │   ├── EskfFilter.kt
│   │       │   │   │   └── FusionState.kt
│   │       │   │   ├── mapmatch/      # Offline OpenStreetMap HMM Viterbi Map-Matcher
│   │       │   │   │   ├── MapMatcher.kt
│   │       │   │   │   └── OsmGraphIndex.kt
│   │       │   │   ├── sensor/        # 100 Hz IMU & GNSS Foreground Service
│   │       │   │   │   ├── ImuSensorManager.kt
│   │       │   │   │   ├── GnssMonitor.kt
│   │       │   │   │   └── NavigationService.kt
│   │       │   │   ├── ui/            # Jetpack Compose 10 Hz Real-Time Navigation UI
│   │       │   │   │   ├── theme/     # Material 3 Dark Automotive Theme
│   │       │   │   │   ├── components/# Beacons, Telemetry Gauges, Simulation Toggles
│   │       │   │   │   ├── MapScreen.kt
│   │       │   │   │   └── NavigationViewModel.kt
│   │       │   │   └── NaviCoreApp.kt # Application entry point & dependency injection
│   │       │   ├── assets/            # Offline Assets bundled in APK
│   │       │   │   ├── models/        # navicore_odometer_int8.tflite
│   │       │   │   ├── tiles/         # city_offline.mbtiles / pmtiles
│   │       │   │   └── styles/        # dark_navigation_style.json
│   │       │   └── res/               # Android drawables, strings, layouts
│   │       └── test/                  # Kotlin Unit Tests & ESKF Trajectory Tests
│   ├── build.gradle.kts               # Project-level build script
│   └── settings.gradle.kts
│
├── core_cpp/                          # Standalone C++20 Native Fusion & Kalman Math Engine
│   ├── CMakeLists.txt                 # Standalone CMake build (supports Linux, Windows, Android)
│   ├── include/navicore/
│   │   ├── types.hpp                  # State vectors, ImuSample, GnssFix, Matrix definitions
│   │   ├── fusion/
│   │   │   └── eskf.hpp               # 15-State Error-State Kalman Filter Header
│   │   ├── calibration/
│   │   │   └── auto_calib.hpp         # LPF Gravity + Horizontal PCA Calibration Header
│   │   ├── zupt/
│   │   │   └── nhc_zupt.hpp           # Non-Holonomic Constraints & ZUPT Trigger Header
│   │   ├── mapmatch/
│   │   │   └── hmm_matcher.hpp        # HMM Viterbi Map-Matching Header
│   │   ├── routing/
│   │   │   └── rerouting_engine.hpp   # Dynamic rerouting engine header
│   │   ├── sensors/
│   │   │   └── barometer_tracker.hpp  # Barometer elevation tracking header
│   │   └── vehicle/
│   │       └── vehicle_profile_manager.hpp # Vehicle profiles and kinematic limits
│   ├── src/
│   │   ├── fusion/
│   │   │   └── eskf.cpp               # ESKF Predict, Update, Covariance Reset Implementation
│   │   ├── calibration/
│   │   │   └── auto_calib.cpp         # Dynamic Mounting Calibration Implementation
│   │   ├── zupt/
│   │   │   └── nhc_zupt.cpp           # NHC Pseudo-measurements & FFT Spectral ZUPT
│   │   └── mapmatch/
│   │       └── hmm_matcher.cpp        # Road network topological snapping
│   ├── tests/                         # GoogleTest Native C++ Test Suite
│   │   └── fusion/
│   │       └── test_eskf.cpp          # Test 15-state convergence on simulated trajectories
│   └── third_party/                   # Vendored Header-Only Dependencies
│       ├── eigen3/                    # Eigen 3.4.0 Linear Algebra (MPL2)
│       └── nanoflann/                 # Fast KD-Tree / R-Tree for spatial search (BSD)
│
├── ml_pipeline/                       # Machine Learning Training, Evaluation & INT8 Export
│   ├── requirements.txt               # Pinned Python dependencies (PyTorch, SciPy, ONNX, TFLite)
│   ├── pyproject.toml                 # Ruff, Black, Mypy configurations
│   ├── config/
│   │   └── training_config.yaml       # Hyperparameters, window_len, learning rate, loss weights
│   ├── src/
│   │   ├── dataset/
│   │   │   ├── iovnbd_loader.py       # IO-VNBD Dataset loader & time-synchronizer
│   │   │   ├── preprocessing.py       # Decimation, dynamic gravity removal, normalization
│   │   │   ├── augmentations.py       # Pothole shocks, idle harmonics, orientation jitter
│   │   │   └── windowing.py           # Configurable sliding-window generator
│   │   ├── models/
│   │   │   ├── tcn_odometer.py        # 1D-TCN with Dilated Convolutions & SE Attention
│   │   │   ├── loss.py                # Multi-task Gaussian NLL + ZUPT BCE loss
│   │   │   └── spectral_zupt.py       # FFT spectral energy band ratio extractor
│   │   ├── training/
│   │   │   ├── train.py               # End-to-end training harness with MLflow/WandB logging
│   │   │   └── evaluate.py            # Evaluation on held-out test splits, RMSE & ZUPT metrics
│   │   └── export/
│   │       └── quantize_tflite.py     # INT8 Post-Training Quantization & TFLite packaging
│   └── tests/
│       ├── test_model_io_contract.py  # Verifies exported TFLite matches Android input tensor
│       └── test_preprocessing.py      # Verifies zero-phase decimation & coordinate normalization
│
├── ros2_node/                         # Enterprise Edge ROS2 Robotics Package (Post-Hackathon)
│   ├── CMakeLists.txt                 # ROS2 ament_cmake build script
│   ├── package.xml                    # ROS2 package manifest (rclcpp, nav_msgs, sensor_msgs)
│   ├── include/navicore_ros2/
│   │   └── navicore_node.hpp          # ROS2 Node subscriber/publisher definitions
│   └── src/
│       └── navicore_node.cpp          # Streams /navicore/odom and /navicore/pose to RViz
│
├── data/                              # Local Dataset & Map Storage (Git-Ignored)
│   ├── raw_iovnbd/                    # Downloaded IO-VNBD dataset CSV files
│   └── osm_tiles/                     # Processed .mbtiles / .pmtiles and vector styles
│
├── scripts/                           # Developer Automation & Build Helper Scripts
│   ├── download_iovnbd.py             # Automates downloading & verifying dataset
│   ├── process_osm_tiles.sh           # Converts OSM .osm.pbf to offline .mbtiles
│   └── run_benchmarks.py              # Executes end-to-end drift simulation on recorded logs
│
├── .gitignore                         # Comprehensive gitignore for Android, C++, Python & Datasets
├── LICENSE                            # Apache 2.0 Open-Source License
└── README.md                          # Master Project Overview & Quickstart
```

---

## 2. Module Responsibilities & Boundary Contracts

### 2.1 Separation of Concerns

```
+---------------------------------------------------------------------------------------------------+
|                                 MODULE RESPONSIBILITY BREAKDOWN                                   |
+-------------------+------------------------------------------+------------------------------------+
| Directory         | Primary Responsibility                   | Technology / Language              |
+-------------------+------------------------------------------+------------------------------------+
| `android_app/`    | Mobile App UI, Sensor Services, Map View | Kotlin, Jetpack Compose, MapLibre  |
| `core_cpp/`       | 15-State ESKF, PCA Calibration, Math     | C++20, Eigen 3.4, GoogleTest       |
| `ml_pipeline/`    | AI Training, INT8 Quantization, Eval     | Python 3.10+, PyTorch, TFLite PTQ  |
| `ros2_node/`      | Robotics Edge Node & RViz Odometry       | C++20, ROS 2 Humble (rclcpp)       |
| `docs/`           | PRD, TRD, Architecture, Security, Guides | Markdown (GitHub Flavored)         |
+-------------------+------------------------------------------+------------------------------------+
```

---

## 3. Data Flow Across the Architecture

```
[Phone Hardware Sensors]
         │  (100 Hz Accel, Gyro, Magnetometer)
         ▼
[android_app/.../sensor/ImuSensorManager.kt]
         │  (DirectByteBuffer: Zero-Copy Pointer)
         ▼
[android_app/src/main/cpp/jni_bridge.cpp]
         │  (Direct float* pass-through)
         ▼
[core_cpp/src/calibration/auto_calib.cpp] ────────► Computes Dynamic R_b^v (LPF + PCA)
         │  (Calibrated Vehicle-Frame IMU Tensor)
         ▼
[android_app/.../odometer/VirtualOdometer.kt] (TFLite INT8 on NNAPI)
         │  (Outputs: Vx, σ_v^2, P(stopped))
         ▼
[core_cpp/src/fusion/eskf.cpp] ──────────────► Fuses (Vx + NHC + ZUPT + GNSS Biases)
         │  (Outputs: FusionState: Lat, Lon, Speed, Heading, Covariance)
         ▼
[core_cpp/src/mapmatch/hmm_matcher.cpp] ────────► Snaps position to OSM road centerline
         │  (Snapped Coordinate)
         ▼
[android_app/.../ui/MapScreen.kt] ─────► 10 Hz Smooth Map Marker on MapLibre Canvas
```
