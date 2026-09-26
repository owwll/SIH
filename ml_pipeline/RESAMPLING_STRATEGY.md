# NaviCore AI: Resampling & Decimation Strategy

**Smart India Hackathon 2026** | **Problem Statement ID**: 260168  
**Module**: Machine Learning Pipeline (`ml_pipeline/`)  
**Specification Reference**: PRD §2.1, TRD §2.1, TRD §4.1, CODE-STYLE.md §4  

---

## 1. Measured Empirical Rate Verification

### 1.1 Empirical Dataset Analysis
Analysis of the raw benchmark files from Coventry University's IO-VNBD dataset (`data/sample_iovnbd/S-S1.csv` and `data/sample_iovnbd/V-S1.csv`) was performed across all timestamp delta intervals:

| Dataset File | Stream Source | Total Rows | Duration (s) | Timestamp Column | Measured $\Delta t$ | Measured Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `S-S1.csv` | Android Smartphone (IMU + GPS) | 1,200 | 119.90 s | `Time Since Start` (ms) | Exactly 100.00 ms | **10.00 Hz** |
| `V-S1.csv` | Racelogic VBOX CAN Bus Ground Truth | 1,200 | 119.90 s | `Time since start of day` (s) | Exactly 0.1000 s | **10.00 Hz** |

* Empirical Timestamp Delta Range: $\min = 0.1000\text{ s}$, $\max = 0.1000\text{ s}$, $\text{std} = 0.0000\text{ s}$.
* Nominal `Sampleperiod` column in CAN data explicitly records: `0.1` seconds.
* Verification Conclusion: The empirical sampling rate is **10.00 Hz**, confirming the publication specification (Onyekpe et al., *IO-VNBD: Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning*, 2021).

### 1.2 Rate Discrepancy Note
Early architectural scoping documents assumed a 100 Hz training ingestion rate. The empirical verification proves that all ground-truth CAN wheel speed and synchronized smartphone IMU data are provided at 10 Hz. Proceeding with an un-decimated 100 Hz assumption would cause a 10x temporal scaling distortion in the 1D-TCN receptive field.

---

## 2. On-Device 100 Hz $\rightarrow$ 10 Hz Downsampling Pipeline

On-device Android sensor hardware (`ImuSensorManager.kt`) samples the physical accelerometer and gyroscope at **100 Hz** (`SENSOR_DELAY_FASTEST` / `SENSOR_DELAY_GAME`) to support low-latency high-bandwidth orientation tracking and shock detection.

To feed the neural virtual odometer (`VirtualOdometer.kt` / TFLite runtime), the 100 Hz stream must be decimated by a factor of:

$$M = \frac{f_{\text{device}}}{f_{\text{target}}} = \frac{100\text{ Hz}}{10\text{ Hz}} = 10$$

### 2.1 Nyquist Limit & Spectral Aliasing Threat
* At target rate $f_{\text{target}} = 10\text{ Hz}$, the Nyquist folding frequency is:
  $$f_{\text{Nyquist}} = \frac{f_{\text{target}}}{2} = 5.0\text{ Hz}$$
* High-frequency noise components present in automotive environments:
  1. **Engine idle vibration harmonics**: 15–30 Hz (e.g., 4-cylinder passenger sedan idle at 750–900 RPM produces fundamental combustion frequencies around 22–25 Hz).
  2. **Chassis resonance and road roughness**: 10–45 Hz.
* **Risk of naive subsampling**: Taking every 10th sample without filtering causes 22 Hz engine vibrations to alias to $|22 - 2 \cdot 10| = 2\text{ Hz}$ in the baseband, masquerading as false vehicle longitudinal acceleration and causing odometer drift.

---

## 3. Anti-Aliasing Filter Implementation

### 3.1 Two-Stage Preprocessing Filter
Before decimation, an anti-aliasing low-pass filter (LPF) is applied to all 6 IMU axes:

1. **Stage 1: Quasi-Static Gravity Isolation (LPF $f_c = 0.5\text{ Hz}$)**
   - Separates constant $1g$ ($9.81\text{ m/s}^2$) gravity component from dynamic accelerations:
     $$\alpha = \frac{\Delta t_{\text{raw}}}{\text{RC} + \Delta t_{\text{raw}}} = \frac{0.01}{0.318 + 0.01} \approx 0.0305$$
     $$\mathbf{g}_k = \alpha \mathbf{a}_k + (1 - \alpha) \mathbf{g}_{k-1}$$
     $$\mathbf{a}_{\text{dyn}, k} = \mathbf{a}_k - \mathbf{g}_k$$

2. **Stage 2: Anti-Aliasing Decimation Filter**
   - **Offline / Python Pipeline (`IMUPreprocessor.decimate_stream`)**:
     Zero-phase boxcar finite-impulse response (FIR) averaging over the $M=10$ sample decimation window:
     $$\bar{x}[m] = \frac{1}{M} \sum_{k=0}^{M-1} x[m \cdot M + k]$$
     This provides a $\text{sinc}$ frequency response with transmission zeroes at integer multiples of 10 Hz, maximally suppressing the 20 Hz and 30 Hz harmonic bands.
   - **Alternative / Scipy Baseline**:
     8th-order Chebyshev Type I low-pass filter with cutoff $f_c = 4.0\text{ Hz}$ (attenuation $> 20\text{ dB}$ at $5.0\text{ Hz}$ Nyquist) applied via `scipy.signal.decimate`.
   - **Real-Time Edge Implementation (Android NDK)**:
     Moving-average circular accumulator over the 100 Hz buffer, emitted at each 100 ms boundary ($M=10$ samples).

### 3.2 Latency and Phase Delay
* The $M=10$ boxcar filter has a constant group delay:
  $$\tau_g = \frac{M - 1}{2 \cdot f_{\text{device}}} = \frac{9}{200} = 45\text{ ms}$$
* Because the TFLite virtual odometer operates on 1.0-second sliding windows (10 samples @ 10 Hz) updated at 10 Hz, this 45 ms anti-aliasing delay is well within the 100 ms inference deadline and eliminates aliasing artifacts.

---

## 4. Configuration Contract (docs/CODE-STYLE.md §4)

To prevent hardcoding of rates across the codebase:
* In `ml_pipeline/config/training_config.yaml`:
  ```yaml
  data:
    training_rate_hz: 10          # Native IO-VNBD rate
    device_sampling_rate_hz: 100  # On-device hardware rate
    window_len_samples: 10        # 1.0 second window at 10 Hz
    stride_samples: 1             # 10 Hz inference stride
  ```
* All loader functions in `iovnbd_loader.py` dynamically resolve `training_rate_hz` from `training_config.yaml` rather than using bare integer literals (`10` or `100`).
