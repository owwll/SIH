"""
NaviCore AI: Official IO-VNBD Dataset Loader & Resampling Synchronizer.
Parses Smartphone (S-*.csv) and Vehicle CAN (V-*.csv) ground truth schemas,
enforces configuration-driven sampling rates, and verifies zero dropped rows.

Smart India Hackathon 2026 | Problem Statement ID: 260168
Specification Reference: docs/CODE-STYLE.md §4, docs/PRD.md §2.1, docs/TRD.md §2.1
"""

import os
import sys
import csv
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
import numpy as np

# Reconfigure console streams for Windows compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    Dataset = object


def find_training_config_path() -> Path:
    """Resolves training_config.yaml path relative to this file or repository root."""
    current_file = Path(__file__).resolve()
    # ml_pipeline/src/dataset/iovnbd_loader.py -> ml_pipeline/config/training_config.yaml
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
    Never relies on magic sample rates.
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
        }
    }


def get_configured_training_rate_hz(config_path: Optional[str] = None) -> int:
    """Returns the training sample rate defined in configuration (default: 10 Hz)."""
    cfg = load_training_config(config_path)
    return int(cfg.get("data", {}).get("training_rate_hz", 10))


def get_configured_window_len_samples(config_path: Optional[str] = None) -> int:
    """Returns sliding window length in samples defined in configuration."""
    cfg = load_training_config(config_path)
    return int(cfg.get("data", {}).get("window_len_samples", 10))


def get_configured_stride_samples(config_path: Optional[str] = None) -> int:
    """Returns inference stride in samples defined in configuration."""
    cfg = load_training_config(config_path)
    return int(cfg.get("data", {}).get("stride_samples", 1))


class IOVNBDLoader:
    """
    Production loader for IO-VNBD dataset files (Smartphone S-*.csv and CAN V-*.csv).
    Enforces configuration-driven sample rate and validates schema completeness.
    """

    def __init__(
        self,
        sample_rate_hz: Optional[int] = None,
        stationary_threshold_mps: float = 0.05,
        config_path: Optional[str] = None,
    ):
        """
        :param sample_rate_hz: Sample rate in Hz. If None, dynamically resolved from training_config.yaml.
        :param stationary_threshold_mps: Speed threshold (m/s) below which vehicle is classified as stationary (ZUPT).
        :param config_path: Optional explicit path to training_config.yaml.
        """
        self.config = load_training_config(config_path)
        self.sample_rate_hz = (
            sample_rate_hz
            if sample_rate_hz is not None
            else get_configured_training_rate_hz(config_path)
        )
        self.dt = 1.0 / float(self.sample_rate_hz)
        self.stationary_threshold_mps = stationary_threshold_mps

    def parse_smartphone_csv(self, filepath: str) -> Dict[str, Any]:
        """
        Parses 24-column S-*.csv smartphone file end-to-end with zero dropped rows.
        Columns:
        [0: lat, 1: lon, 2: alt, 3: speed_kmh, 4: acc_m, 5: ori_deg, 6: sats, 7: time_ms, 8: date,
         9: ax, 10: ay, 11: az, 12: grav_x, 13: grav_y, 14: grav_z,
         15: gyro_yaw(gz), 16: gyro_pitch(gy), 17: gyro_roll(gx),
         18: mag_x, 19: mag_y, 20: mag_z, 21: ori_yaw, 22: ori_roll, 23: ori_pitch]
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Smartphone data file not found: {filepath}")

        timestamps_ms = []
        imu_6dof = []
        gps_speed_mps = []
        gps_speed_kmh = []
        gps_coords = []
        dropped_rows = 0
        total_data_rows = 0

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row_idx, row in enumerate(reader):
                if not row or all(c.strip() == "" for c in row):
                    continue
                total_data_rows += 1
                try:
                    lat = float(row[0])
                    lon = float(row[1])
                    spd_kmh = float(row[3])
                    t_ms = float(row[7])
                    ax = float(row[9])
                    ay = float(row[10])
                    az = float(row[11])
                    gz = float(row[15])
                    gy = float(row[16])
                    gx = float(row[17])

                    timestamps_ms.append(t_ms)
                    imu_6dof.append([ax, ay, az, gx, gy, gz])
                    gps_speed_kmh.append(spd_kmh)
                    gps_speed_mps.append(spd_kmh / 3.6)
                    gps_coords.append([lat, lon])
                except (ValueError, IndexError) as err:
                    dropped_rows += 1

        timestamps_s = np.array(timestamps_ms, dtype=np.float64) / 1000.0
        time_deltas = np.diff(timestamps_s) if len(timestamps_s) > 1 else np.array([self.dt])
        empirical_rate = 1.0 / np.mean(time_deltas) if len(time_deltas) > 0 else float(self.sample_rate_hz)

        return {
            "filepath": filepath,
            "row_count": len(imu_6dof),
            "total_rows_read": total_data_rows,
            "dropped_rows": dropped_rows,
            "duration_s": float(timestamps_s[-1] - timestamps_s[0]) if len(timestamps_s) > 1 else 0.0,
            "empirical_rate_hz": float(empirical_rate),
            "mean_dt_s": float(np.mean(time_deltas)) if len(time_deltas) > 0 else self.dt,
            "timestamps_s": timestamps_s,
            "imu_6dof": np.array(imu_6dof, dtype=np.float32),
            "gps_speed_mps": np.array(gps_speed_mps, dtype=np.float32),
            "gps_speed_kmh": np.array(gps_speed_kmh, dtype=np.float32),
            "gps_coords": np.array(gps_coords, dtype=np.float64),
            "speed_min_mps": float(np.min(gps_speed_mps)) if gps_speed_mps else 0.0,
            "speed_max_mps": float(np.max(gps_speed_mps)) if gps_speed_mps else 0.0,
            "speed_mean_mps": float(np.mean(gps_speed_mps)) if gps_speed_mps else 0.0,
        }

    def parse_vehicle_can_csv(self, filepath: str) -> Dict[str, Any]:
        """
        Parses 29-column V-*.csv vehicle CAN ground truth file end-to-end with zero dropped rows.
        Columns:
        [0: sats, 1: time_s, 2: lat, 3: lon, 4: vel_kmh, 5: hdg, 6: alt, 7: vert_vel,
         8: sample_period, 9: steer_angle, 10: ws_fl, 11: ws_fr, 12: ws_rl, 13: ws_rr,
         14: yaw_rate, 15: indicated_spd_kmh, 16: long_accel_g, 17: lat_accel_g, ...]
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vehicle CAN data file not found: {filepath}")

        time_s = []
        can_speed_mps = []
        can_speed_kmh = []
        wheel_speeds = []
        yaw_rate = []
        dropped_rows = 0
        total_data_rows = 0

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row_idx, row in enumerate(reader):
                if not row or all(c.strip() == "" for c in row):
                    continue
                total_data_rows += 1
                try:
                    ts = float(row[1])
                    vel_kmh = float(row[4])
                    ws_fl = float(row[10])
                    ws_fr = float(row[11])
                    ws_rl = float(row[12])
                    ws_rr = float(row[13])
                    yr = float(row[14])

                    time_s.append(ts)
                    can_speed_kmh.append(vel_kmh)
                    can_speed_mps.append(vel_kmh / 3.6)
                    wheel_speeds.append([ws_fl, ws_fr, ws_rl, ws_rr])
                    yaw_rate.append(yr)
                except (ValueError, IndexError):
                    dropped_rows += 1

        time_arr = np.array(time_s, dtype=np.float64)
        time_deltas = np.diff(time_arr) if len(time_arr) > 1 else np.array([self.dt])
        empirical_rate = 1.0 / np.mean(time_deltas) if len(time_deltas) > 0 else float(self.sample_rate_hz)

        speeds_mps = np.array(can_speed_mps, dtype=np.float32)
        zupt_stationary = speeds_mps < self.stationary_threshold_mps
        stationary_count = int(np.sum(zupt_stationary))
        moving_count = len(speeds_mps) - stationary_count

        return {
            "filepath": filepath,
            "row_count": len(can_speed_mps),
            "total_rows_read": total_data_rows,
            "dropped_rows": dropped_rows,
            "duration_s": float(time_arr[-1] - time_arr[0]) if len(time_arr) > 1 else 0.0,
            "empirical_rate_hz": float(empirical_rate),
            "mean_dt_s": float(np.mean(time_deltas)) if len(time_deltas) > 0 else self.dt,
            "time_s": time_arr,
            "can_speed_mps": speeds_mps,
            "can_speed_kmh": np.array(can_speed_kmh, dtype=np.float32),
            "wheel_speeds": np.array(wheel_speeds, dtype=np.float32),
            "yaw_rate": np.array(yaw_rate, dtype=np.float32),
            "zupt_stationary": zupt_stationary.astype(np.float32),
            "speed_min_mps": float(np.min(speeds_mps)) if len(speeds_mps) > 0 else 0.0,
            "speed_max_mps": float(np.max(speeds_mps)) if len(speeds_mps) > 0 else 0.0,
            "speed_mean_mps": float(np.mean(speeds_mps)) if len(speeds_mps) > 0 else 0.0,
            "stationary_count": stationary_count,
            "moving_count": moving_count,
            "stationary_pct": (stationary_count / len(speeds_mps) * 100.0) if len(speeds_mps) > 0 else 0.0,
            "moving_pct": (moving_count / len(speeds_mps) * 100.0) if len(speeds_mps) > 0 else 0.0,
        }

    def load_and_summarize(self, s_filepath: str, v_filepath: str) -> Dict[str, Any]:
        """
        Parses both files end-to-end, asserts zero dropped rows, and prints complete statistical summary.
        """
        s_res = self.parse_smartphone_csv(s_filepath)
        v_res = self.parse_vehicle_can_csv(v_filepath)

        # Assert no dropped rows
        assert s_res["dropped_rows"] == 0, f"S file had {s_res['dropped_rows']} dropped rows"
        assert v_res["dropped_rows"] == 0, f"V file had {v_res['dropped_rows']} dropped rows"
        assert s_res["row_count"] == v_res["row_count"], (
            f"Row count mismatch: S={s_res['row_count']}, V={v_res['row_count']}"
        )

        n_rows = s_res["row_count"]
        duration_s = max(s_res["duration_s"], v_res["duration_s"])

        print("=" * 80)
        print(" 🚗 NAVICORE AI: IO-VNBD DATASET LOADER & VERIFICATION AUDIT")
        print(" Smart India Hackathon 2026 | PS 260168 | Team @enigm@ (132834)")
        print("=" * 80)
        print(f" [✓] Config Training Rate  : {self.sample_rate_hz} Hz (from training_config.yaml)")
        print(f" [✓] Smartphone Data File  : {s_filepath}")
        print(f" [✓] Vehicle CAN Data File : {v_filepath}")
        print("-" * 80)
        print(" 1. INGESTION INTEGRITY:")
        print(f"    • Smartphone (S-S1) Rows Parsed : {s_res['row_count']:,} / {s_res['total_rows_read']:,} (Dropped: {s_res['dropped_rows']})")
        print(f"    • Vehicle CAN (V-S1) Rows Parsed: {v_res['row_count']:,} / {v_res['total_rows_read']:,} (Dropped: {v_res['dropped_rows']})")
        print(f"    • Ingestion Status              : ✅ 100% CLEAN (0 Dropped Rows)")
        print("-" * 80)
        print(" 2. TEMPORAL DURATION & EMPIRICAL RATE:")
        print(f"    • Total Trajectory Duration     : {duration_s:.2f} seconds ({duration_s / 60.0:.2f} minutes)")
        print(f"    • Smartphone Empirical Rate     : {s_res['empirical_rate_hz']:.2f} Hz (mean dt = {s_res['mean_dt_s']*1000.0:.1f} ms)")
        print(f"    • Vehicle CAN Empirical Rate    : {v_res['empirical_rate_hz']:.2f} Hz (mean dt = {v_res['mean_dt_s']*1000.0:.1f} ms)")
        print("-" * 80)
        print(" 3. SPEED RANGE & DYNAMICS:")
        print(f"    • CAN Ground Truth Speed (m/s)  : [{v_res['speed_min_mps']:.2f} m/s, {v_res['speed_max_mps']:.2f} m/s] (Mean: {v_res['speed_mean_mps']:.2f} m/s)")
        print(f"    • CAN Ground Truth Speed (km/h) : [{v_res['speed_min_mps']*3.6:.2f} km/h, {v_res['speed_max_mps']*3.6:.2f} km/h] (Mean: {v_res['speed_mean_mps']*3.6:.2f} km/h)")
        print(f"    • Phone GPS Speed (m/s)         : [{s_res['speed_min_mps']:.2f} m/s, {s_res['speed_max_mps']:.2f} m/s] (Mean: {s_res['speed_mean_mps']:.2f} m/s)")
        print("-" * 80)
        print(" 4. ZUPT / STATIONARY CLASS BALANCE (Threshold < 0.05 m/s):")
        print(f"    • Stationary (ZUPT = 1) Samples : {v_res['stationary_count']:,} ({v_res['stationary_pct']:.2f}%)")
        print(f"    • Moving     (ZUPT = 0) Samples : {v_res['moving_count']:,} ({v_res['moving_pct']:.2f}%)")
        print(f"    • Class Balance Ratio           : {v_res['stationary_pct']:.1f}% Stopped / {v_res['moving_pct']:.1f}% Moving")
        print("=" * 80)

        return {
            "row_count": n_rows,
            "duration_s": duration_s,
            "speed_range_mps": (v_res["speed_min_mps"], v_res["speed_max_mps"]),
            "speed_range_kmh": (v_res["speed_min_mps"] * 3.6, v_res["speed_max_mps"] * 3.6),
            "class_balance": {
                "stationary_count": v_res["stationary_count"],
                "stationary_pct": v_res["stationary_pct"],
                "moving_count": v_res["moving_count"],
                "moving_pct": v_res["moving_pct"],
            },
            "smartphone_data": s_res,
            "vehicle_can_data": v_res,
        }


class SyntheticDriveGenerator:
    """
    Standalone vehicle drive generator producing 6-DOF IMU data and ground truth.
    Rate is initialized from training configuration YAML, never hardcoded.
    """

    def __init__(self, sample_rate_hz: Optional[int] = None, config_path: Optional[str] = None):
        self.sample_rate_hz = (
            sample_rate_hz
            if sample_rate_hz is not None
            else get_configured_training_rate_hz(config_path)
        )
        self.dt = 1.0 / float(self.sample_rate_hz)

    def generate_tunnel_blackout_run(self, duration_sec: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
        steps = int(duration_sec * self.sample_rate_hz)
        imu = np.zeros((steps, 6), dtype=np.float32)
        gt = np.zeros((steps, 3), dtype=np.float32)

        vx = 16.6
        for i in range(steps):
            imu[i, 0] = np.random.normal(0.0, 0.05)  # Ax
            imu[i, 1] = np.random.normal(0.0, 0.02)  # Ay
            imu[i, 2] = 9.81 + np.random.normal(0.0, 0.05)  # Az
            imu[i, 3] = np.random.normal(0.0, 0.01)  # Gx
            imu[i, 4] = np.random.normal(0.0, 0.01)  # Gy
            imu[i, 5] = 0.005  # Gz bias
            gt[i, 0] = vx * (i * self.dt)
            gt[i, 1] = 0.0
            gt[i, 2] = vx
        return imu, gt


class SyntheticDriveDataset(Dataset):
    """
    Synthetic dataset for neural virtual odometer validation.
    Window length, stride, and sample rate read from config values, never hardcoded.
    """

    def __init__(
        self,
        num_trajectories: int = 50,
        traj_duration_s: float = 60.0,
        sample_rate_hz: Optional[int] = None,
        window_len: Optional[int] = None,
        stride: Optional[int] = None,
        config_path: Optional[str] = None,
    ):
        self.sample_rate_hz = (
            sample_rate_hz
            if sample_rate_hz is not None
            else get_configured_training_rate_hz(config_path)
        )
        self.window_len = (
            window_len
            if window_len is not None
            else get_configured_window_len_samples(config_path)
        )
        self.stride = (
            stride
            if stride is not None
            else get_configured_stride_samples(config_path)
        )
        self.dt = 1.0 / float(self.sample_rate_hz)

        self.windows: List[np.ndarray] = []
        self.target_vx: List[float] = []
        self.target_zupt: List[float] = []

        self._generate_dataset(num_trajectories, traj_duration_s)

    def _generate_dataset(self, num_trajectories: int, duration_s: float):
        steps = int(duration_s * self.sample_rate_hz)
        time_arr = np.linspace(0, duration_s, steps)

        for _ in range(num_trajectories):
            speed_profile = np.zeros(steps, dtype=np.float32)
            current_v = 0.0
            t = 0
            while t < steps:
                seg_type = np.random.choice(["accel", "cruise", "decel", "stop"], p=[0.25, 0.35, 0.2, 0.2])
                seg_len = np.random.randint(int(3 * self.sample_rate_hz), int(12 * self.sample_rate_hz))
                end_t = min(steps, t + seg_len)

                if seg_type == "accel":
                    target_v = np.random.uniform(8.0, 25.0)
                    accel_rate = np.random.uniform(1.0, 2.5)
                    for k in range(t, end_t):
                        current_v = min(target_v, current_v + accel_rate * self.dt)
                        speed_profile[k] = current_v
                elif seg_type == "cruise":
                    for k in range(t, end_t):
                        speed_profile[k] = current_v + np.random.normal(0.0, 0.1)
                elif seg_type == "decel":
                    decel_rate = np.random.uniform(1.5, 3.5)
                    for k in range(t, end_t):
                        current_v = max(0.0, current_v - decel_rate * self.dt)
                        speed_profile[k] = current_v
                elif seg_type == "stop":
                    current_v = 0.0
                    for k in range(t, end_t):
                        speed_profile[k] = 0.0

                t = end_t

            accel_x = np.gradient(speed_profile, self.dt)
            imu_stream = np.zeros((steps, 6), dtype=np.float32)
            imu_stream[:, 0] = accel_x + np.random.normal(0.0, 0.15, steps)

            yaw_rate = np.zeros(steps, dtype=np.float32)
            if np.random.rand() > 0.5:
                turn_start = np.random.randint(10, max(11, steps - 30))
                turn_dur = np.random.randint(20, 50)
                yaw_rate[turn_start:turn_start + turn_dur] = np.random.uniform(-0.15, 0.15)
                imu_stream[:, 1] = speed_profile * yaw_rate + np.random.normal(0.0, 0.1, steps)
            else:
                imu_stream[:, 1] = np.random.normal(0.0, 0.08, steps)

            imu_stream[:, 2] = 9.81 + np.random.normal(0.0, 0.25, steps)
            imu_stream[:, 3] = np.random.normal(0.0, 0.02, steps)
            imu_stream[:, 4] = np.random.normal(0.0, 0.02, steps)
            imu_stream[:, 5] = yaw_rate + np.random.normal(0.0, 0.015, steps)

            pothole_count = np.random.randint(1, 6)
            for _ in range(pothole_count):
                p_idx = np.random.randint(0, steps)
                imu_stream[p_idx, 0] += np.random.uniform(-3.0, 3.0)
                imu_stream[p_idx, 2] += np.random.uniform(4.0, 10.0)

            is_stopped = speed_profile < 0.05
            for k in np.where(is_stopped)[0]:
                idle_vib = 0.4 * np.sin(2.0 * np.pi * 22.0 * time_arr[k])
                imu_stream[k, 0] += idle_vib
                imu_stream[k, 2] += idle_vib * 1.5

            num_w = (steps - self.window_len) // self.stride + 1
            for w in range(num_w):
                s_idx = w * self.stride
                e_idx = s_idx + self.window_len
                self.windows.append(imu_stream[s_idx:e_idx].T)
                target_v = speed_profile[e_idx - 1]
                self.target_vx.append(target_v)
                self.target_zupt.append(1.0 if target_v < 0.05 else 0.0)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        if _HAS_TORCH:
            x = torch.tensor(self.windows[idx], dtype=torch.float32)
            y_vx = torch.tensor([self.target_vx[idx]], dtype=torch.float32)
            y_zupt = torch.tensor([self.target_zupt[idx]], dtype=torch.float32)
            return x, y_vx, y_zupt
        return self.windows[idx], np.array([self.target_vx[idx]], dtype=np.float32), np.array([self.target_zupt[idx]], dtype=np.float32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NaviCore AI: IO-VNBD Loader & Verification")
    parser.add_argument("--s_file", default="data/sample_iovnbd/S-S1.csv", help="Path to S-*.csv file")
    parser.add_argument("--v_file", default="data/sample_iovnbd/V-S1.csv", help="Path to V-*.csv file")
    parser.add_argument("--config", default=None, help="Path to training_config.yaml")
    args = parser.parse_args()

    loader = IOVNBDLoader(config_path=args.config)
    loader.load_and_summarize(args.s_file, args.v_file)
