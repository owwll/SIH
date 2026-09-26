import os
import sys
import pytest
from pathlib import Path

# Add ml_pipeline and ml_pipeline/src to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

from src.dataset.iovnbd_loader import (
    IOVNBDLoader,
    get_configured_training_rate_hz,
    load_training_config,
)


def test_config_rate_resolution():
    """Verify that sample rate is read dynamically from training_config.yaml."""
    cfg = load_training_config()
    assert "data" in cfg
    assert "training_rate_hz" in cfg["data"]
    rate = get_configured_training_rate_hz()
    assert rate == cfg["data"]["training_rate_hz"]
    assert rate == 10


def test_parse_smartphone_csv_end_to_end():
    """Verify S-S1.csv parses with zero dropped rows and 10 Hz rate."""
    s_path = "data/sample_iovnbd/S-S1.csv"
    if not os.path.exists(s_path):
        pytest.skip(f"Dataset file {s_path} not found")

    loader = IOVNBDLoader()
    res = loader.parse_smartphone_csv(s_path)
    assert res["dropped_rows"] == 0
    assert res["row_count"] == 1200
    assert abs(res["empirical_rate_hz"] - 10.0) < 0.05
    assert abs(res["duration_s"] - 119.9) < 0.2
    assert res["imu_6dof"].shape == (1200, 6)


def test_parse_vehicle_can_csv_end_to_end():
    """Verify V-S1.csv parses with zero dropped rows, speed ranges, and ZUPT balance."""
    v_path = "data/sample_iovnbd/V-S1.csv"
    if not os.path.exists(v_path):
        pytest.skip(f"Dataset file {v_path} not found")

    loader = IOVNBDLoader()
    res = loader.parse_vehicle_can_csv(v_path)
    assert res["dropped_rows"] == 0
    assert res["row_count"] == 1200
    assert abs(res["empirical_rate_hz"] - 10.0) < 0.05
    assert abs(res["duration_s"] - 119.9) < 0.2
    assert res["stationary_count"] == 301
    assert res["moving_count"] == 899
    assert abs(res["stationary_pct"] - 25.08) < 0.1


def test_load_and_summarize():
    """Verify full summary execution over both files."""
    s_path = "data/sample_iovnbd/S-S1.csv"
    v_path = "data/sample_iovnbd/V-S1.csv"
    if not os.path.exists(s_path) or not os.path.exists(v_path):
        pytest.skip("Sample dataset files not found")

    loader = IOVNBDLoader()
    summary = loader.load_and_summarize(s_path, v_path)
    assert summary["row_count"] == 1200
    assert summary["class_balance"]["stationary_count"] == 301
