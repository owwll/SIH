#!/usr/bin/env python3
"""
=============================================================================
NAVICORE AI: DATASET AUGMENTATIONS & STRESS INJECTION ENGINE
Smart India Hackathon 2026 | Problem Statement ID: 260168
Theme: Smart Vehicles | Team: @enigm@ (Team ID: 132834)
=============================================================================
Implements physically grounded data augmentations for 1D-TCN Virtual Odometer:
1. Synthetic Pothole/Shock Injection: Short, high-magnitude vertical acceleration impulse
2. Engine-Idle Harmonic Vibration: 15-30 Hz engine combustion vibration during stops (ZUPT robustness)
3. Gaussian Sensor Noise: White Gaussian noise across all 6 IMU channels
4. Random Mount Orientation Jitter: +/-15 deg SO(3) 3D rotational perturbation

Every augmentation is independently toggleable and parameterized via training_config.yaml.
Zero magic sample rate literals (docs/CODE-STYLE.md §4).
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


def find_training_config_path() -> Path:
    """Resolves training_config.yaml path relative to this file or repository root."""
    current_file = Path(__file__).resolve()
    candidate_1 = current_file.parent.parent.parent / "config" / "training_config.yaml"
    if candidate_1.exists():
        return candidate_1
    candidate_2 = Path("ml_pipeline/config/training_config.yaml").resolve()
    if candidate_2.exists():
        return candidate_2
    candidate_3 = Path("../config/training_config.yaml").resolve()
    if candidate_3.exists():
        return candidate_3
    return candidate_1


def load_training_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads training configuration YAML file per docs/CODE-STYLE.md §4.
    Never relies on magic sample rates or hardcoded thresholds.
    """
    path = Path(config_path).resolve() if config_path else find_training_config_path()
    if path.exists() and _HAS_YAML:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            if isinstance(cfg, dict):
                return cfg

    # Fallback to standard config schema if YAML is missing
    return {
        "data": {
            "training_rate_hz": 10,
            "device_sampling_rate_hz": 100,
            "window_len_samples": 10,
            "stride_samples": 1,
        },
        "augmentations": {
            "enabled": True,
            "pothole_shock": {
                "enabled": True,
                "shock_intensity_g": 3.5,
                "duration_samples": 2,
                "probability": 0.35,
            },
            "engine_idle": {
                "enabled": True,
                "idle_freq_hz": 22.0,
                "min_freq_hz": 15.0,
                "max_freq_hz": 30.0,
                "amplitude_mps2": 0.8,
            },
            "gaussian_noise": {
                "enabled": True,
                "accel_std": 0.05,
                "gyro_std": 0.005,
            },
            "orientation_jitter": {
                "enabled": True,
                "max_angle_deg": 15.0,
            },
        },
    }


class SensorAugmenter:
    """
    Multi-sensor stress injection and physical augmentation engine.
    Fully parameterized and configurable via training_config.yaml.
    """

    def __init__(
        self,
        sample_rate_hz: Optional[float] = None,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        """
        :param sample_rate_hz: Sampling rate in Hz. If None, dynamically resolved from training_config.yaml.
        :param config_path: Path to training_config.yaml.
        :param config: Direct config dictionary override.
        :param seed: PRNG seed for deterministic augmentations.
        """
        if config is not None:
            self.cfg = config
        else:
            self.cfg = load_training_config(config_path)

        data_cfg = self.cfg.get("data", {})
        aug_cfg = self.cfg.get("augmentations", {})

        # Sampling rate from config or parameter, never hardcoded
        if sample_rate_hz is not None:
            self.fs = float(sample_rate_hz)
        else:
            self.fs = float(data_cfg.get("training_rate_hz", 10))

        self.rng = np.random.default_rng(seed)

        # Augmentation parameters
        self.aug_enabled = bool(aug_cfg.get("enabled", True))

        # 1. Pothole / Shock parameters
        pothole_cfg = aug_cfg.get("pothole_shock", {})
        self.pothole_enabled = bool(pothole_cfg.get("enabled", True))
        self.pothole_intensity_g = float(pothole_cfg.get("shock_intensity_g", 3.5))
        self.pothole_duration_samples = int(pothole_cfg.get("duration_samples", 2))
        self.pothole_prob = float(pothole_cfg.get("probability", 0.35))

        # 2. Engine idle parameters
        idle_cfg = aug_cfg.get("engine_idle", {})
        self.idle_enabled = bool(idle_cfg.get("enabled", True))
        self.idle_freq_hz = float(idle_cfg.get("idle_freq_hz", 22.0))
        self.idle_min_freq_hz = float(idle_cfg.get("min_freq_hz", 15.0))
        self.idle_max_freq_hz = float(idle_cfg.get("max_freq_hz", 30.0))
        self.idle_amplitude_mps2 = float(idle_cfg.get("amplitude_mps2", 0.8))

        # 3. Gaussian sensor noise parameters
        noise_cfg = aug_cfg.get("gaussian_noise", {})
        self.noise_enabled = bool(noise_cfg.get("enabled", True))
        self.accel_std = float(noise_cfg.get("accel_std", 0.05))
        self.gyro_std = float(noise_cfg.get("gyro_std", 0.005))

        # 4. Orientation jitter parameters
        jitter_cfg = aug_cfg.get("orientation_jitter", {})
        self.jitter_enabled = bool(jitter_cfg.get("enabled", True))
        self.jitter_max_angle_deg = float(jitter_cfg.get("max_angle_deg", 15.0))

    def _ensure_time_first(self, imu_array: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Normalizes input array to time-first format [N_samples, 6].
        Returns (array_n_6, was_channels_first).
        """
        arr = np.asarray(imu_array, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D IMU array, got shape {arr.shape}")

        if arr.shape[0] == 6 and arr.shape[1] != 6:
            # Channels-first [6, N] -> transpose to [N, 6]
            return arr.T.copy(), True
        elif arr.shape[1] == 6:
            # Time-first [N, 6]
            return arr.copy(), False
        else:
            raise ValueError(
                f"Expected one dimension to be 6 (ax, ay, az, gx, gy, gz), got shape {arr.shape}"
            )

    def _restore_shape(self, imu_array: np.ndarray, was_channels_first: bool) -> np.ndarray:
        """Restores original shape if input was channels-first [6, N]."""
        return imu_array.T if was_channels_first else imu_array

    def add_gaussian_noise(
        self,
        imu_array: np.ndarray,
        accel_std: Optional[float] = None,
        gyro_std: Optional[float] = None,
    ) -> np.ndarray:
        """
        Adds independent zero-mean Gaussian sensor noise across all 6 IMU channels.
        Models MEMS accelerometer and gyroscope thermal noise and digitizer jitter.

        :param imu_array: IMU array of shape [N, 6] or [6, N]
        :param accel_std: Noise std on accelerometer channels in m/s^2 (default from config: 0.05)
        :param gyro_std: Noise std on gyroscope channels in rad/s (default from config: 0.005)
        :return: Augmented IMU array matching input shape
        """
        arr, was_cf = self._ensure_time_first(imu_array)
        n_samples = arr.shape[0]

        a_std = self.accel_std if accel_std is None else float(accel_std)
        g_std = self.gyro_std if gyro_std is None else float(gyro_std)

        noise_acc = self.rng.normal(0.0, a_std, size=(n_samples, 3))
        noise_gyr = self.rng.normal(0.0, g_std, size=(n_samples, 3))

        arr[:, 0:3] += noise_acc.astype(np.float32)
        arr[:, 3:6] += noise_gyr.astype(np.float32)

        return self._restore_shape(arr, was_cf)

    def inject_pothole_shock(
        self,
        imu_array: np.ndarray,
        shock_intensity_g: Optional[float] = None,
        duration_samples: Optional[int] = None,
        axis: int = 2,
    ) -> np.ndarray:
        """
        Injects a short, high-magnitude impulsive shock simulating a vehicle hitting a pothole or bump.
        Impulse follows a half-sine acceleration profile: a(t) = A * sin(pi * t / T).
        Guaranteed to never silently no-op, even on short windows.

        :param imu_array: IMU array of shape [N, 6] or [6, N]
        :param shock_intensity_g: Peak shock amplitude in Gs (default from config: 3.5 G)
        :param duration_samples: Duration of shock in samples (default from config: 2)
        :param axis: Acceleration axis for primary impulse (default: 2 -> Z-axis vertical)
        :return: Augmented IMU array matching input shape
        """
        arr, was_cf = self._ensure_time_first(imu_array)
        n_samples = arr.shape[0]

        intensity_g = self.pothole_intensity_g if shock_intensity_g is None else float(shock_intensity_g)
        req_duration = self.pothole_duration_samples if duration_samples is None else int(duration_samples)

        # Never silently no-op: bound duration between 1 and n_samples
        effective_duration = max(1, min(n_samples, req_duration))
        max_start = n_samples - effective_duration

        start_idx = int(self.rng.integers(0, max_start + 1)) if max_start > 0 else 0
        shock_amplitude_mps2 = intensity_g * 9.80665
        # Midpoint sampling ensures non-zero, symmetric pulse for any N >= 1, reaching peak shock amplitude
        t_pulse = (np.arange(effective_duration, dtype=np.float32) + 0.5) * (np.pi / float(effective_duration))
        pulse_shape = np.sin(t_pulse)
        shock_profile = (pulse_shape / np.max(pulse_shape)) * shock_amplitude_mps2

        # Primary vertical impulse
        arr[start_idx : start_idx + effective_duration, axis] += shock_profile
        # Secondary longitudinal/pitch reaction (realistic suspension rebound ~ 15% amplitude)
        rebound_axis = 0 if axis == 2 else 2
        arr[start_idx : start_idx + effective_duration, rebound_axis] -= 0.15 * shock_profile

        return self._restore_shape(arr, was_cf)

    def inject_engine_idle_harmonics(
        self,
        imu_array: np.ndarray,
        idle_freq_hz: Optional[float] = None,
        amplitude_mps2: Optional[float] = None,
        randomize_freq: bool = True,
    ) -> np.ndarray:
        """
        Injects internal combustion engine idle harmonic vibration in the 15-30 Hz band.
        Trains the neural network and ZUPT detector to classify vehicle as stationary despite
        chassis vibration caused by an idling engine at traffic stops.

        :param imu_array: IMU array of shape [N, 6] or [6, N]
        :param idle_freq_hz: Specific vibration frequency. If None, samples uniformly in [15.0, 30.0] Hz.
        :param amplitude_mps2: Peak body acceleration vibration amplitude (default: 0.8 m/s^2)
        :param randomize_freq: If True and idle_freq_hz is None, randomly selects frequency in [min, max] band
        :return: Augmented IMU array matching input shape
        """
        arr, was_cf = self._ensure_time_first(imu_array)
        n_samples = arr.shape[0]

        if idle_freq_hz is not None:
            freq = float(idle_freq_hz)
        elif randomize_freq:
            freq = float(self.rng.uniform(self.idle_min_freq_hz, self.idle_max_freq_hz))
        else:
            freq = self.idle_freq_hz

        amp = self.idle_amplitude_mps2 if amplitude_mps2 is None else float(amplitude_mps2)

        # Time vector based on configured sample rate (fs), SI units
        t = np.arange(n_samples, dtype=np.float32) / float(self.fs)
        phase_z = float(self.rng.uniform(0.0, 2.0 * np.pi))
        phase_xy = float(self.rng.uniform(0.0, 2.0 * np.pi))

        # Vertical primary vibration + subtle 3D engine mount vibration coupling
        vibe_z = amp * np.sin(2.0 * np.pi * freq * t + phase_z)
        vibe_x = (0.25 * amp) * np.sin(2.0 * np.pi * freq * t + phase_xy)
        vibe_y = (0.20 * amp) * np.cos(2.0 * np.pi * freq * t + phase_xy)

        arr[:, 2] += vibe_z.astype(np.float32)
        arr[:, 0] += vibe_x.astype(np.float32)
        arr[:, 1] += vibe_y.astype(np.float32)

        return self._restore_shape(arr, was_cf)

    def apply_orientation_jitter(
        self,
        imu_array: np.ndarray,
        max_angle_deg: Optional[float] = None,
    ) -> np.ndarray:
        """
        Applies a random 3D rotation perturbation in SO(3) within +/-max_angle_deg.
        Simulates slight phone mount vibrations, bumps, or loose cradle positioning,
        evaluating the robustness of neural odometer predictions independent of auto-calibration.

        :param imu_array: IMU array of shape [N, 6] or [6, N]
        :param max_angle_deg: Maximum rotation perturbation angle per axis (default: 15.0 deg per spec)
        :return: Augmented IMU array matching input shape
        """
        arr, was_cf = self._ensure_time_first(imu_array)

        max_deg = self.jitter_max_angle_deg if max_angle_deg is None else float(max_angle_deg)
        angles_deg = self.rng.uniform(-max_deg, max_deg, size=3)
        angles_rad = np.radians(angles_deg)

        roll, pitch, yaw = angles_rad
        cx, cy, cz = np.cos([roll, pitch, yaw])
        sx, sy, sz = np.sin([roll, pitch, yaw])

        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        # Orthonormal rotation matrix in SO(3): det(R) = 1, R^T R = I
        R = (Rz @ Ry @ Rx).astype(np.float32)

        # Rotate 3D accelerometer channels [ax, ay, az]
        arr[:, 0:3] = (R @ arr[:, 0:3].T).T
        # Rotate 3D gyroscope channels [gx, gy, gz]
        arr[:, 3:6] = (R @ arr[:, 3:6].T).T

        return self._restore_shape(arr, was_cf)

    def augment_window(
        self,
        window_imu: np.ndarray,
        is_stationary: bool = False,
        enable_pothole: Optional[bool] = None,
        enable_engine_idle: Optional[bool] = None,
        enable_gaussian_noise: Optional[bool] = None,
        enable_orientation_jitter: Optional[bool] = None,
        pothole_shock_intensity_g: Optional[float] = None,
        pothole_duration_samples: Optional[int] = None,
        idle_freq_hz: Optional[float] = None,
        idle_amplitude_mps2: Optional[float] = None,
        accel_noise_std: Optional[float] = None,
        gyro_noise_std: Optional[float] = None,
        orientation_jitter_max_deg: Optional[float] = None,
    ) -> np.ndarray:
        """
        Applies a configurable, independently toggleable pipeline of physical augmentations
        to a single sliding window [N, 6] or [6, N].

        :param window_imu: IMU sliding window array
        :param is_stationary: True if vehicle is stationary (ZUPT active)
        :param enable_pothole: Explicit toggle for pothole shock (defaults to config and moving state)
        :param enable_engine_idle: Explicit toggle for engine idle (defaults to config and stationary state)
        :param enable_gaussian_noise: Explicit toggle for Gaussian sensor noise
        :param enable_orientation_jitter: Explicit toggle for mount orientation jitter
        :return: Fully augmented IMU window
        """
        out = window_imu.copy()

        # 1. Gaussian sensor noise
        do_noise = self.noise_enabled if enable_gaussian_noise is None else bool(enable_gaussian_noise)
        if do_noise:
            out = self.add_gaussian_noise(out, accel_std=accel_noise_std, gyro_std=gyro_noise_std)

        # 2. Orientation jitter (+/- 15 deg)
        do_jitter = self.jitter_enabled if enable_orientation_jitter is None else bool(enable_orientation_jitter)
        if do_jitter:
            out = self.apply_orientation_jitter(out, max_angle_deg=orientation_jitter_max_deg)

        # 3. Engine-idle harmonics (15-30 Hz)
        if enable_engine_idle is not None:
            do_idle = bool(enable_engine_idle)
        else:
            do_idle = self.idle_enabled and is_stationary

        if do_idle:
            out = self.inject_engine_idle_harmonics(
                out, idle_freq_hz=idle_freq_hz, amplitude_mps2=idle_amplitude_mps2
            )

        # 4. Pothole / Shock injection
        if enable_pothole is not None:
            do_pothole = bool(enable_pothole)
        else:
            do_pothole = self.pothole_enabled and (not is_stationary) and (self.rng.random() < self.pothole_prob)

        if do_pothole:
            out = self.inject_pothole_shock(
                out, shock_intensity_g=pothole_shock_intensity_g, duration_samples=pothole_duration_samples
            )

        return out
