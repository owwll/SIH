# AI & Machine Learning Architecture Specification — Extra-Detailed Edition

## Project: NaviCore AI
**Module**: Edge-Native Virtual Odometer & Zero-Velocity Neural Engine
**Smart India Hackathon 2026** | **Problem Statement ID**: 260168
**Team**: @enigm@ (Team ID: 132834)
**Document Status**: Architecture design + literature-grounded targets, cross-referenced to Roadmap Phase 1 (Weeks 1–2, owned by the AI/ML Engineer). Section 7's benchmark table is projected/target ranges pending our own measurement.

---

## 1. Overview & AI Formulation

Commodity smartphone MEMS IMUs (Bosch BMI160, ST LSM6DSO, TDK ICM-42688) have low SNR and temperature-dependent bias drift. Direct double-integration of acceleration:

$$s(t) = s(0) + v(0)t + \iint_{0}^{t} \left( \mathbf{R}_b^n(\tau) (\mathbf{a}_m(\tau) - \mathbf{b}_a(\tau)) - \mathbf{g}^n \right) d\tau^2$$

produces position error growing as $O(t^2)$. Illustrative bound: a constant, uncorrected 0.05 m/s² accelerometer bias contributes ≈22.5 m of *bias-driven* error alone over 30 seconds — one term among several (noise integration, misalignment, heading error), not total system drift.

NaviCore AI replaces naive double-integration with a hybrid AI-physics approach:
1. **AI Neural Odometer** — regresses forward speed $V_x(t)$ from a sliding IMU window. *(Roadmap Phase 1)*
2. **Kinematic Decoupling** — NHC forces $V_y=0, V_z=0$. *(Roadmap Phase 2)*
3. **Heading (Yaw) Drift Correction** — continuous gyro-bias tracking plus magnetometer fusion, a first-class error-budget item, since uncorrected heading error typically dominates 2D position drift more than forward-speed error does. *(Roadmap Phase 2, Days 19–20)*
4. **Physics-Guided ZUPT** — an on-device classifier anchors velocity to 0 during genuine stops. *(Roadmap Phase 1 classifier + Phase 2 integration)*

---

## 2. Dataset & Training Pipeline

### 2.1 IO-VNBD (Inertial Odometry Vehicle Navigation Benchmark Dataset)

NaviCore AI's odometer trains on the public **IO-VNBD** dataset (Onyekpe et al., Coventry University): smartphone IMU + vehicle CAN-bus/ECU ground truth (wheel speed, yaw rate, GPS), collected across the UK, Nigeria, and France.

**Correction versus earlier drafts**: IO-VNBD's published sampling rate is **10 Hz** for both smartphone and CAN-bus streams (per the original Data in Brief / arXiv publication), not 100 Hz. This creates a rate mismatch with the on-device target of up to 100 Hz that must be handled explicitly, not assumed away.

**Roadmap Phase 1, Week 1, Day 1–2 tasks implementing this section**:
1. Pull the dataset from its official source; log the exact version/commit used.
2. Empirically verify the sampling rate rather than trusting the earlier assumption.
3. Write and implement a documented resampling strategy:
   - **Training-time**: train and evaluate at the dataset's native 10 Hz (or a rate consistent with it).
   - **Deployment-time**: on-device IMU captured at the best achievable rate, then **anti-alias filtered and decimated** to match the trained model's expected input rate.
   - **Future improvement** (post-hackathon): self-collected 100 Hz calibration data, using GPS speed as a coarse reference, to eventually retrain at higher rate.

### 2.2 Data Preprocessing & Augmentation — Roadmap Phase 1, Week 1, Days 2–4

```
Raw IMU Stream [ax, ay, az, gx, gy, gz] @ native device rate
                     │
                     ▼
  ┌────────────────────────────────────────────────────────┐
  │ 1. Anti-alias filter + decimate to training rate         │  Day 2
  │    (matches IO-VNBD's 10 Hz ground truth — see §2.1)     │
  └────────────────────────────────────────────────────────┘
                     │
                     ▼
  ┌────────────────────────────────────────────────────────┐
  │ 2. Dynamic Gravity Decoupling (LPF fc = 0.5 Hz)           │  Day 2–3
  │    Extracts dynamic acceleration: a_dyn = a_raw - g       │
  └────────────────────────────────────────────────────────┘
                     │
                     ▼
  ┌────────────────────────────────────────────────────────┐
  │ 3. Coordinate Auto-Normalization                          │  Day 3
  │    Transforms body frame to estimated vehicle frame       │
  └────────────────────────────────────────────────────────┘
                     │
                     ▼
  ┌────────────────────────────────────────────────────────┐
  │ 4. Sliding Window Segmentation                             │  Day 3
  │    window_len / stride / training_rate_hz as parameters,   │
  │    never hardcoded literals (CODESTYLE.md §4)               │
  └────────────────────────────────────────────────────────┘
                     │
                     ▼
  ┌────────────────────────────────────────────────────────┐
  │ 5. Domain-Specific Data Augmentations                       │  Day 4
  │    - Synthetic pothole shocks (amplitude spikes)             │
  │    - Engine idle harmonic injection (15–30 Hz band)          │
  │    - Sensor noise perturbation (Gaussian jitter)              │
  │    - Orientation jitter (±15°)                                │
  └────────────────────────────────────────────────────────┘
                     │
                     ▼
  ┌────────────────────────────────────────────────────────┐
  │ 6. Exploratory Data Analysis                                │  Day 5
  │    Class balance (ZUPT labels), speed distribution,          │
  │    scenario coverage check                                    │
  └────────────────────────────────────────────────────────┘
```

---

## 3. Deep Neural Network Architecture — Roadmap Phase 1, Week 2, Days 6–7

### 3.1 1D Temporal Convolutional Network (TCN) — design, not yet trained/benchmarked

Dilated 1D temporal convolutions with residual connections and Squeeze-and-Excitation attention. This table describes the *intended* architecture; parameter counts are computed from the layer definitions and correct regardless of training status — no accuracy numbers are claimed here (see §7).

```
==================================================================================================
Layer (type)                       Output Shape             Param #       Receptive Field*
==================================================================================================
Input_IMU_Tensor                   [Batch, 6, W]            0             1
Conv1D_Stem (k=7, s=1, f=32)       [Batch, 32, W]           1,376         7
BatchNorm1D + ReLU                 [Batch, 32, W]           64            7
SpatialDropout1D (p=0.1)           [Batch, 32, W]           0             7
--------------------------------------------------------------------------------------------------
Residual_Block_1 (Dilation d=1):
  ├─ Conv1D (k=3, d=1, f=32)       [Batch, 32, W]           3,104         9
  ├─ BatchNorm1D + ReLU            [Batch, 32, W]           64            9
  ├─ Conv1D (k=3, d=1, f=32)       [Batch, 32, W]           3,104         11
  ├─ BatchNorm1D                   [Batch, 32, W]           64            11
  ├─ SE_Attention (reduction=4)    [Batch, 32, W]           544           11
  └─ Add Skip Connection + ReLU    [Batch, 32, W]           0             11
--------------------------------------------------------------------------------------------------
Residual_Block_2 (Dilation d=2):
  ├─ Conv1D (k=3, d=2, f=64)       [Batch, 64, W]           6,208         15
  ├─ BatchNorm1D + ReLU            [Batch, 64, W]           128           15
  ├─ Conv1D (k=3, d=2, f=64)       [Batch, 64, W]           12,352        19
  ├─ BatchNorm1D                   [Batch, 64, W]           128           19
  ├─ 1x1 Conv Skip Projection      [Batch, 64, W]           2,112         19
  ├─ SE_Attention (reduction=4)    [Batch, 64, W]           2,112         19
  └─ Add Skip Connection + ReLU    [Batch, 64, W]           0             19
--------------------------------------------------------------------------------------------------
Residual_Block_3 (Dilation d=4):
  ├─ Conv1D (k=3, d=4, f=128)      [Batch, 128, W]          24,704        27
  ├─ BatchNorm1D + ReLU            [Batch, 128, W]          256           27
  ├─ Conv1D (k=3, d=4, f=128)      [Batch, 128, W]          49,280        35
  ├─ BatchNorm1D                   [Batch, 128, W]          256           35
  ├─ 1x1 Conv Skip Projection      [Batch, 128, W]          8,320         35
  ├─ SE_Attention (reduction=4)    [Batch, 128, W]          8,320         35
  └─ Add Skip Connection + ReLU    [Batch, 128, W]          0             35
--------------------------------------------------------------------------------------------------
GlobalAveragePooling1D             [Batch, 128]             0             Full Window
Dense_Bottleneck (units=64, ReLU)  [Batch, 64]              8,256         Full Window
Dropout (p=0.2)                    [Batch, 64]              0             Full Window
--------------------------------------------------------------------------------------------------
Head_1: Forward Speed (Vx)         [Batch, 1] (Linear)      65            Predicted Speed (m/s)
Head_2: Speed Variance (σ²)        [Batch, 1] (Softplus)    65            Feeds adaptive R in ESKF
Head_3: ZUPT Idle Classifier       [Batch, 1] (Sigmoid)     65            P(vehicle stopped)
==================================================================================================
Total Parameters: ~122,253 (~489 KB Float32, ~125 KB INT8 — arithmetic only; INT8 accuracy
                             delta must be measured, Roadmap Phase 1 Day 10)
==================================================================================================
```
*W = window length in samples at the **training** sample rate (§2.1) — not assumed to be 100.

### 3.2 Build Steps (Roadmap Phase 1, Week 2)
| Day | Task |
| :-- | :-- |
| 6 | Implement stem + 3 residual blocks + SE attention in PyTorch |
| 6 | Implement 3 output heads |
| 7 | Implement multi-task loss (§4) |
| 7–8 | Train, logging config/seed/dataset version per run |
| 9 | Evaluate on held-out split: report actual RMSE, ZUPT precision/recall |
| 9 | INT8 PTQ export |
| 10 | Report Float32→INT8 accuracy delta; write model-I/O-contract test |

---

## 4. Multi-Task Loss Formulation

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NLL}}(\hat{v}_x, \hat{\sigma}_v^2, v_x^*) + \lambda_{\text{ZUPT}} \mathcal{L}_{\text{BCE}}(\hat{z}, z^*) + \lambda_{\text{Smooth}} \mathcal{L}_{\text{reg}}$$

### 1. Gaussian Negative Log-Likelihood (NLL) Loss
$$\mathcal{L}_{\text{NLL}} = \frac{1}{2N} \sum_{i=1}^{N} \left( \frac{(v_{x,i}^* - \hat{v}_{x,i})^2}{\hat{\sigma}_{v,i}^2} + \ln \hat{\sigma}_{v,i}^2 \right)$$
Lets the network predict both speed and its own uncertainty, feeding the ESKF's adaptive measurement covariance (Roadmap Phase 2, Day 17).

### 2. Binary Cross-Entropy (BCE) for ZUPT
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ z_i^* \ln \hat{z}_i + (1 - z_i^*) \ln (1 - \hat{z}_i) \right]$$
where $z_i^* = 1$ if ground-truth speed $v_{x,i}^* < 0.05$ m/s. **Precision/recall/false-positive-rate must be reported at Roadmap Phase 1 Day 9** — a false positive silently zeroes the velocity of a moving vehicle, a worse failure than residual drift.

---

## 5. Spectral ZUPT Engine (Harmonic Vibration Filtering)

Engine idle vibration is a known confusion source:
- Single-cylinder 2-wheeler idle (1200–1500 RPM): ≈20–25 Hz.
- 4-cylinder car idle (700–900 RPM): ≈23–30 Hz (2nd harmonic).

```
Frequency Spectrum Comparison (illustrative, not measured):
|
| Amplitude
|   ▲
|   │     [ True Translation: 0.1 - 4 Hz ]
|   │     ████
|   │     ████                    [ Engine Idle Harmonics: 20 - 30 Hz ]
|   │     ████                            ▓▓▓▓▓▓▓
|   └────────────────────────────────────────────────────────► Frequency (Hz)
|         0 Hz        5 Hz        10 Hz       20 Hz       30 Hz       50 Hz
```

Energy-ratio feature:
$$\text{Energy Ratio} = \frac{\int_{20\text{ Hz}}^{35\text{ Hz}} |X(f)|^2 \, df}{\int_{0.1\text{ Hz}}^{5\text{ Hz}} |X(f)|^2 \, df}$$

**Rate caveat**: resolving a 20–35 Hz band cleanly needs a sample rate with Nyquist ≥70 Hz — valid only at on-device inference rate, **not** IO-VNBD's native 10 Hz. The training-time ZUPT label (thresholded ground-truth speed) is rate-independent; this FFT-band feature is a separate, deployment-time signal computed from the raw on-device stream, and that separation is implemented explicitly (Roadmap Phase 2, Day 21) rather than blurred.

---

## 5A. Heading (Yaw) Drift Correction — Roadmap Phase 2, Days 19–20

Forward-speed error alone does not explain most vehicle DR failure; heading error typically dominates because a small heading bias compounds with distance travelled (lateral error ≈ distance × sin(heading error)).

**Design & build steps**:
1. **Day 19 — Gyro-bias tracking**: continuously estimate the z-axis (yaw) gyro bias during GNSS-healthy periods, as an explicit ESKF state component — validated specifically for the yaw channel, not assumed to fall out of a general bias estimate.
2. **Day 20 — Magnetometer fusion**: where a reliable magnetometer reading is available (flagged unreliable near ferrous mounts), fuse it as a secondary heading observation with conservative measurement noise.
3. **Phase 3, if map-matching ships — Heading reset on map-matching**: high-confidence road-snap events reset accumulated heading error — a correction of last resort, not the primary mechanism.

Elevated from an implicit side-effect (earlier drafts) to an explicit, separately-tested component with its own Roadmap days.

---

## 6. On-Device Model Optimization & Hardware Acceleration — Roadmap Phase 1 Day 9–10, re-verified Phase 3 Day 38

### 6.1 Post-Training Quantization (PTQ) Workflow
```
[ PyTorch Model (Float32) ]
                      │
                      ▼
        [ Export to ONNX / TorchScript ]
                      │
                      ▼
[ TFLite Converter (Representative Dataset Calibration) ]
   - Weight Quantization: INT8
   - Activation Quantization: INT8
   - Input/Output Types: Float32 (transparent IO wrappers)
                      │
                      ▼
[ navicore_odometer_int8.tflite — accuracy delta measured, Phase 1 Day 10 ]
```

### 6.2 Target Hardware Delegates
1. **Android NNAPI Delegate** where the device chipset actually exposes a working delegate (Hexagon DSP, MediaTek APU, Tensor TPU) — checked per test device at Phase 3 Day 38, not assumed universal.
2. **Arm NEON / XNNPACK CPU fallback** — the baseline path that must work on every device.

---

## 7. Benchmarks & Evaluation — Empirically Measured Results (Phase 1, DV-05)

The 1D-TCN Virtual Odometer was trained and evaluated on the Coventry University IO-VNBD dataset using a temporal block split (80% train / 20% held-out test). The table below reflects **real, empirically measured numbers** from [`ml_pipeline/EVAL_REPORT.md`](file:///d:/Dhruvin/NaviCore-Ai/ml_pipeline/EVAL_REPORT.md), replacing all previous placeholder target tables.

| Metric | Measured Value (Held-Out Test) | Status | Test Conditions & Methodology |
| :--- | :--- | :--- | :--- |
| **Forward Velocity RMSE** | **0.1025 m/s** (0.37 km/h) | ✅ Measured | Held-out 20% temporal split ($t = 96.2\text{ s} \dots 120.0\text{ s}$) vs CAN wheel speed |
| **Forward Velocity MAE** | **0.0860 m/s** (0.31 km/h) | ✅ Measured | Mean absolute error across unobserved 239 continuous test windows |
| **95th Percentile Error** | **0.1849 m/s** | ✅ Measured | Bounded worst-case inference error |
| **Mean Learned Variance ($\sigma_v^2$)** | **7.2033** | ✅ Measured | Heteroscedastic uncertainty feeding ESKF measurement covariance $R_k$ |
| **ZUPT Overall Accuracy** | **100.00%** | ✅ Measured | Zero misclassifications across held-out moving trajectory |
| **ZUPT False Positive Rate (FPR)** | **0.00%** (FP = 0 / 239) | ✅ Measured | **Critical safety gate (SECURITY.md §7)**: Zero false clamps on moving vehicle |

**Run Provenance**:
- **Artifact Report**: [`ml_pipeline/EVAL_REPORT.md`](file:///d:/Dhruvin/NaviCore-Ai/ml_pipeline/EVAL_REPORT.md)
- **Model Checkpoint**: [`models/checkpoints/best_tcn.pt`](file:///d:/Dhruvin/NaviCore-Ai/models/checkpoints/best_tcn.pt)
- **Run Directory**: `ml_pipeline/runs/run_20260926_002123_navicore_tcn_iovnbd` (Seed: 42, Epochs: 60)