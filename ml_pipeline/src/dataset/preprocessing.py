"""
=============================================================================
NAVICORE AI: DATA PREPROCESSING, DECIMATION, & GRAVITY DECOUPLING MODULE
Smart India Hackathon 2026 | Problem Statement ID: 260168
Theme: Smart Vehicles | Team: @enigm@ (Team ID: 132834)
=============================================================================
Handles:
1. Anti-aliasing decimation (100 Hz phone hardware -> 10 Hz training rate)
2. Quasi-static gravity decoupling via 1st-order IIR Low-Pass Filter (LPF fc=0.5 Hz)
3. Sliding window tensor generation for 1D-TCN Virtual Odometer inference

CRITICAL PIPELINE ORDER (docs/TRD.md §3.2, docs/CODE-STYLE.md §2):
Gravity decoupling MUST execute on the continuous IMU stream BEFORE windowing,
NEVER on isolated sliced windows.
Filtering isolated 1.0 s (10-sample) windows causes severe filter initialization
transients (tau = 1 / (2*pi*0.5) ~ 0.318 s, settling time ~ 1.0-1.5 s), corrupting
up to 100% of the window. Continuous filtering prior to window slicing maintains
smooth filter state memory across window boundaries with zero edge transients.
"""

import numpy as np
from typing import Tuple, Optional


class IMUPreprocessor:
    def __init__(
        self,
        raw_sampling_rate_hz: int = 100,
        target_sampling_rate_hz: int = 10,
        lpf_cutoff_hz: float = 0.5,
    ):
        """
        :param raw_sampling_rate_hz: Hardware sensor sample rate (e.g., 100 Hz).
        :param target_sampling_rate_hz: ML model input rate (e.g., 10 Hz matching IO-VNBD).
        :param lpf_cutoff_hz: Low-pass filter cutoff for static gravity isolation (default: 0.5 Hz).
        """
        self.raw_rate = int(raw_sampling_rate_hz)
        self.target_rate = int(target_sampling_rate_hz)
        self.lpf_cutoff_hz = float(lpf_cutoff_hz)
        self.decimation_factor = max(1, self.raw_rate // self.target_rate)

        # Exponential smoothing factor for discrete 1st-order LPF:
        # alpha = dt / (RC + dt), where RC = 1 / (2 * pi * fc)
        # Note: Filter operates on the decimated rate (target_rate) if decimated,
        # or raw_rate if decimation is bypassed.
        dt = 1.0 / float(self.target_rate)
        rc = 1.0 / (2.0 * np.pi * self.lpf_cutoff_hz)
        self.alpha = dt / (rc + dt)

        self.gravity_estimate: Optional[np.ndarray] = None

    def reset_filter(self):
        """Resets the gravity estimate state for new independent streams."""
        self.gravity_estimate = None

    def isolate_gravity(self, accel_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Splits raw continuous accelerometer stream into quasi-static gravity and dynamic body acceleration.
        Implemented as a 1st-order discrete exponential low-pass filter (fc = 0.5 Hz).
        MUST be called on continuous streams before window slicing.

        :param accel_xyz: Array of shape [N, 3] or [3]
        :return: (dynamic_accel, gravity_vector) matching input shape
        """
        accel = np.atleast_2d(accel_xyz)
        n_samples = accel.shape[0]
        dynamic_accel = np.zeros_like(accel, dtype=np.float32)
        gravity = np.zeros_like(accel, dtype=np.float32)

        if self.gravity_estimate is None:
            self.gravity_estimate = np.copy(accel[0]).astype(np.float32)

        for i in range(n_samples):
            self.gravity_estimate = (
                self.alpha * accel[i] + (1.0 - self.alpha) * self.gravity_estimate
            )
            gravity[i] = self.gravity_estimate
            dynamic_accel[i] = accel[i] - self.gravity_estimate

        return np.squeeze(dynamic_accel), np.squeeze(gravity)

    def decimate_stream(self, imu_6dof: np.ndarray) -> np.ndarray:
        """
        Anti-aliasing downsampling of 6-DOF stream [ax, ay, az, gx, gy, gz].
        Applies a box-car moving average filter over decimation_factor samples.

        :param imu_6dof: Array of shape [N, 6]
        :return: Decimated array of shape [N // decimation_factor, 6]
        """
        if self.decimation_factor <= 1:
            return imu_6dof

        # Simple box-car averaging filter for anti-aliasing
        n_trimmed = (len(imu_6dof) // self.decimation_factor) * self.decimation_factor
        trimmed = imu_6dof[:n_trimmed]
        reshaped = trimmed.reshape(-1, self.decimation_factor, 6)
        return np.mean(reshaped, axis=1)

    def create_sliding_windows(
        self,
        imu_stream: np.ndarray,
        window_size: int = 10,
        stride: int = 1,
    ) -> np.ndarray:
        """
        Constructs rolling sliding window tensors for neural network inference.
        CRITICAL: Input imu_stream should already have gravity decoupled.

        :param imu_stream: Array of shape [Total_Samples, 6]
        :param window_size: Number of time-steps per window (e.g., 10 steps @ 10 Hz = 1.0 s).
        :param stride: Window shift step (e.g., 1 step = 10 Hz inference rate).
        :return: 3D tensor of shape [Num_Windows, 6, window_size] (Channels-First)
        """
        num_samples = len(imu_stream)
        if num_samples < window_size:
            return np.empty((0, 6, window_size), dtype=np.float32)

        num_windows = (num_samples - window_size) // stride + 1
        windows = np.zeros((num_windows, 6, window_size), dtype=np.float32)

        for i in range(num_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            # Transpose to Channels-First [6, window_size] for PyTorch Conv1D
            windows[i] = imu_stream[start_idx:end_idx].T

        return windows

    def preprocess_stream(
        self,
        imu_6dof: np.ndarray,
        decouple_gravity: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Executes full signal preprocessing on a continuous 6-DOF IMU stream in the
        strictly required architectural order:
        Step 1: Anti-aliasing decimation (if raw_rate > target_rate)
        Step 2: Continuous gravity decoupling (LPF fc=0.5 Hz) BEFORE windowing
        
        :param imu_6dof: Raw continuous 6-DOF IMU stream [N, 6]
        :param decouple_gravity: If True, subtracts quasi-static gravity from accelerometer
        :return: (processed_stream_6dof, gravity_stream_3d)
        """
        # Step 1: Anti-aliasing decimation
        decimated = self.decimate_stream(imu_6dof)

        # Step 2: Gravity decoupling on the continuous decimated stream
        if decouple_gravity:
            dynamic_accel, gravity = self.isolate_gravity(decimated[:, 0:3])
            # Combine dynamic acceleration with raw gyroscopes
            processed = np.hstack([dynamic_accel, decimated[:, 3:6]])
            return processed, gravity
        else:
            return decimated, None

    def process_and_window(
        self,
        imu_6dof: np.ndarray,
        window_size: int = 10,
        stride: int = 1,
        decouple_gravity: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        End-to-end preprocessing pipeline enforcing:
        Decimation -> Continuous Gravity Decoupling -> Windowing.

        :param imu_6dof: Raw continuous 6-DOF IMU stream [N, 6]
        :param window_size: Window length in samples (at target rate)
        :param stride: Stride in samples (at target rate)
        :param decouple_gravity: Whether to decouple gravity before windowing
        :return: (windows [Num_Windows, 6, window_size], continuous_processed_stream, continuous_gravity)
        """
        processed_stream, gravity = self.preprocess_stream(imu_6dof, decouple_gravity=decouple_gravity)
        windows = self.create_sliding_windows(processed_stream, window_size=window_size, stride=stride)
        return windows, processed_stream, gravity
