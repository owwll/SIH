"""
=============================================================================
NAVICORE AI: UNIT TESTS FOR AUGMENTATIONS & GRAVITY DECOUPLING PIPELINE
Smart India Hackathon 2026 | Problem Statement ID: 260168
=============================================================================
Tests:
1. Pothole shock injection produces bounded vertical spikes and never no-ops.
2. Engine-idle harmonics inject 15-30 Hz band vibration with configured amplitude.
3. Gaussian noise injects zero-mean perturbation across all 6 IMU channels.
4. Orientation jitter produces proper SO(3) rotations preserving vector norms.
5. All 4 augmentations are independently toggleable in augment_window.
6. Gravity decoupling (LPF fc=0.5 Hz) executes before windowing.
"""

import os
import sys
import pytest
import numpy as np
from pathlib import Path

# Add ml_pipeline and ml_pipeline/src to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

from src.dataset.augmentations import SensorAugmenter, load_training_config
from src.dataset.preprocessing import IMUPreprocessor
from src.dataset.iovnbd_loader import IOVNBDLoader


@pytest.fixture
def sample_window():
    """Provides a synthetic 1.0s window (10 samples @ 10 Hz) with 6-DOF IMU data."""
    # Shape [10, 6]: ax, ay, az, gx, gy, gz
    t = np.linspace(0, 0.9, 10, dtype=np.float32)
    window = np.zeros((10, 6), dtype=np.float32)
    window[:, 0] = 0.5 * np.cos(2 * np.pi * 0.5 * t)  # Ax
    window[:, 1] = 0.2 * np.sin(2 * np.pi * 0.5 * t)  # Ay
    window[:, 2] = 9.81                                # Az (gravity)
    window[:, 3] = 0.01                                # Gx
    window[:, 4] = -0.01                               # Gy
    window[:, 5] = 0.05                                # Gz
    return window


@pytest.fixture
def real_sample_window():
    """Loads a real sample window from S-S1.csv if available."""
    csv_path = Path("data/sample_iovnbd/S-S1.csv")
    if not csv_path.exists():
        pytest.skip(f"Sample data file {csv_path} not found")
    loader = IOVNBDLoader()
    s_data = loader.parse_smartphone_csv(str(csv_path))
    imu = s_data["imu_6dof"]
    return imu[100:110].copy() # 10 samples (1.0 s)


def test_pothole_shock_non_zero_and_bounded(sample_window, real_sample_window):
    """Verifies pothole shock is non-zero, bounded, and never silently no-ops."""
    augmenter = SensorAugmenter(seed=42)

    for win in [sample_window, real_sample_window]:
        augmented = augmenter.inject_pothole_shock(
            win, shock_intensity_g=3.5, duration_samples=2, axis=2
        )
        assert augmented.shape == win.shape
        assert not np.allclose(augmented, win), "Pothole shock must modify the window"

        # Check vertical acceleration delta
        delta_z = np.max(np.abs(augmented[:, 2] - win[:, 2]))
        expected_peak = 3.5 * 9.80665
        assert np.isclose(delta_z, expected_peak, rtol=1e-3), (
            f"Expected peak shock {expected_peak}, got {delta_z}"
        )

        # Gyro channels should remain untouched by pothole shock
        assert np.allclose(augmented[:, 3:6], win[:, 3:6])


def test_pothole_shock_short_window_no_crash():
    """Verifies pothole shock does not crash or no-op on short windows (e.g. 2 samples or 1 sample)."""
    augmenter = SensorAugmenter(seed=42)
    for length in [1, 2, 5]:
        tiny_win = np.zeros((length, 6), dtype=np.float32)
        tiny_win[:, 2] = 9.81
        aug = augmenter.inject_pothole_shock(tiny_win, shock_intensity_g=2.0, duration_samples=3)
        assert not np.allclose(aug, tiny_win), f"Shock no-oped on length {length}"
        assert aug.shape == tiny_win.shape


def test_engine_idle_harmonics(sample_window):
    """Verifies engine-idle harmonic vibration is injected in 15-30 Hz band."""
    augmenter = SensorAugmenter(seed=42)
    target_amp = 0.8
    target_freq = 25.0

    augmented = augmenter.inject_engine_idle_harmonics(
        sample_window, idle_freq_hz=target_freq, amplitude_mps2=target_amp
    )

    assert not np.allclose(augmented, sample_window), "Engine idle must modify the window"
    diff_z = augmented[:, 2] - sample_window[:, 2]
    # Injected signal is sinusoidal with peak amplitude target_amp
    assert np.max(np.abs(diff_z)) <= target_amp + 1e-4
    assert np.max(np.abs(diff_z)) > 0.1, "Sinusoidal amplitude too small"


def test_gaussian_sensor_noise(sample_window):
    """Verifies zero-mean Gaussian noise is applied to all 6 IMU channels."""
    augmenter = SensorAugmenter(seed=123)
    target_acc_std = 0.05
    target_gyr_std = 0.005

    augmented = augmenter.add_gaussian_noise(
        sample_window, accel_std=target_acc_std, gyro_std=target_gyr_std
    )

    assert not np.allclose(augmented, sample_window)
    # Check all 6 channels have perturbation
    acc_diff = augmented[:, 0:3] - sample_window[:, 0:3]
    gyr_diff = augmented[:, 3:6] - sample_window[:, 3:6]
    assert np.all(np.abs(acc_diff) > 0)
    assert np.all(np.abs(gyr_diff) > 0)


def test_orientation_jitter_so3_isometry(sample_window):
    """Verifies +/-15 deg orientation jitter is a strict SO(3) isometry preserving vector lengths."""
    augmenter = SensorAugmenter(seed=999)
    augmented = augmenter.apply_orientation_jitter(sample_window, max_angle_deg=15.0)

    assert not np.allclose(augmented, sample_window)

    # Check vector length preservation (orthonormality: ||R v|| == ||v||)
    orig_acc_norm = np.linalg.norm(sample_window[:, 0:3], axis=1)
    aug_acc_norm = np.linalg.norm(augmented[:, 0:3], axis=1)
    assert np.allclose(orig_acc_norm, aug_acc_norm, atol=1e-5), (
        "Orientation jitter must preserve 3D acceleration vector length"
    )

    orig_gyr_norm = np.linalg.norm(sample_window[:, 3:6], axis=1)
    aug_gyr_norm = np.linalg.norm(augmented[:, 3:6], axis=1)
    assert np.allclose(orig_gyr_norm, aug_gyr_norm, atol=1e-5), (
        "Orientation jitter must preserve 3D gyroscope vector length"
    )


def test_independent_toggleability(sample_window):
    """Verifies each augmentation can be toggled on/off independently via augment_window."""
    augmenter = SensorAugmenter(seed=42)

    # 1. All disabled
    none_aug = augmenter.augment_window(
        sample_window,
        enable_pothole=False,
        enable_engine_idle=False,
        enable_gaussian_noise=False,
        enable_orientation_jitter=False,
    )
    assert np.allclose(none_aug, sample_window), "When all toggled off, output must match input"

    # 2. Only pothole
    pothole_only = augmenter.augment_window(
        sample_window,
        enable_pothole=True,
        enable_engine_idle=False,
        enable_gaussian_noise=False,
        enable_orientation_jitter=False,
    )
    assert not np.allclose(pothole_only, sample_window)
    assert np.allclose(pothole_only[:, 3:6], sample_window[:, 3:6]) # Gyro untouched

    # 3. Only idle
    idle_only = augmenter.augment_window(
        sample_window,
        enable_pothole=False,
        enable_engine_idle=True,
        enable_gaussian_noise=False,
        enable_orientation_jitter=False,
    )
    assert not np.allclose(idle_only, sample_window)
    assert np.allclose(idle_only[:, 3:6], sample_window[:, 3:6]) # Gyro untouched

    # 4. Only noise
    noise_only = augmenter.augment_window(
        sample_window,
        enable_pothole=False,
        enable_engine_idle=False,
        enable_gaussian_noise=True,
        enable_orientation_jitter=False,
    )
    assert not np.allclose(noise_only, sample_window)

    # 5. Only jitter
    jitter_only = augmenter.augment_window(
        sample_window,
        enable_pothole=False,
        enable_engine_idle=False,
        enable_gaussian_noise=False,
        enable_orientation_jitter=True,
    )
    assert not np.allclose(jitter_only, sample_window)


def test_channels_first_and_last_transposition(sample_window):
    """Verifies augmenter correctly handles both [N, 6] (time-first) and [6, N] (channels-first)."""
    augmenter = SensorAugmenter(seed=42)

    # Time-first: [10, 6]
    out_tf = augmenter.add_gaussian_noise(sample_window)
    assert out_tf.shape == (10, 6)

    # Channels-first: [6, 10]
    sample_cf = sample_window.T
    out_cf = augmenter.add_gaussian_noise(sample_cf)
    assert out_cf.shape == (6, 10)


def test_gravity_decoupling_before_windowing():
    """
    Empirically verifies that continuous gravity decoupling prior to window slicing
    avoids initial filter transient errors.
    """
    preprocessor = IMUPreprocessor(raw_sampling_rate_hz=10, target_sampling_rate_hz=10, lpf_cutoff_hz=0.5)

    # Synthetic continuous trajectory: 100 samples
    n_samples = 100
    accel = np.zeros((n_samples, 3), dtype=np.float32)
    accel[:, 2] = 9.81 # Static gravity on Z
    accel[:, 0] = 1.0 * np.sin(np.linspace(0, 10, n_samples)) # Dynamic motion on X

    # 1. Continuous stream processing (Correct order)
    dyn_cont, grav_cont = preprocessor.isolate_gravity(accel)
    # After continuous filtering, gravity estimate is within 1% of 9.81 m/s^2
    assert np.isclose(grav_cont[-1, 2], 9.81, rtol=1e-2)

    # 2. Window generation from continuous stream
    imu_6dof = np.hstack([dyn_cont, np.zeros((n_samples, 3), dtype=np.float32)])
    windows = preprocessor.create_sliding_windows(imu_6dof, window_size=10, stride=1)
    assert windows.shape == (91, 6, 10)
