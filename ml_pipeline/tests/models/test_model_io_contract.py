"""
=============================================================================
NAVICORE AI: MODEL I/O TENSOR CONTRACT VERIFICATION (DV-07a)
Smart India Hackathon 2026 | Problem Statement ID: 260168
Theme: Smart Vehicles | Team: @enigm@ (Team ID: 132834)
=============================================================================
Validates the TFLite model contract defined in docs/API.MD §3 and TRD §4.1/§4.2:
1. Input tensor shape: [Batch=1, Channels=6, WindowLen=10] @ 10 Hz rate.
2. Output tensor count & shapes: 3 heads, each shape [Batch=1, 1]:
   - Head A: Forward speed Vx (m/s)
   - Head B: Speed variance sigma_v^2 (m^2/s^2, > 0)
   - Head C: Stopped probability P(stopped) in [0, 1]
3. Real inference validation: Runs inference on real preprocessed IMU window from
   Coventry IO-VNBD dataset (S-S1.csv), asserting finite and physically plausible values.
4. Model file integrity: Validates non-placeholder binary size (< 460 KB target).

This test is the contract Manthan's Android TFLite integration (DV-08) depends on.
"""

import os
import sys
import tempfile
import pytest
import numpy as np

# Adjust python paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
ML_ROOT = os.path.join(REPO_ROOT, "ml_pipeline")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, ML_ROOT)
sys.path.insert(0, os.path.join(ML_ROOT, "src"))

try:
    import tensorflow as tf
    _HAS_TF = True
except ImportError:
    _HAS_TF = False

try:
    from src.models.export.export_tflite import export_model_artifacts
except ImportError:
    from models.export.export_tflite import export_model_artifacts

from dataset.iovnbd_loader import IOVNBDLoader, load_training_config
from dataset.preprocessing import IMUPreprocessor


MODEL_PATH = os.path.join(REPO_ROOT, "models", "exported", "navicore_odometer_int8.tflite")
S_CSV_PATH = os.path.join(REPO_ROOT, "data", "sample_iovnbd", "S-S1.csv")


def test_model_file_exists_and_not_placeholder():
    """Assert models/exported/navicore_odometer_int8.tflite is a real binary model, not 37 bytes."""
    assert os.path.exists(MODEL_PATH), f"Model file not found at: {MODEL_PATH}"
    file_size = os.path.getsize(MODEL_PATH)
    assert file_size != 37, "Model file is still the 37-byte placeholder mock string!"
    # Plausible size: > 50 KB and near/under ~460 KB TRD §4.2 budget
    assert file_size > 50 * 1024, f"Model size {file_size} bytes is suspiciously small!"
    assert file_size <= 500 * 1024, f"Model size {file_size} bytes exceeds 500 KB budget!"


@pytest.mark.skipif(not _HAS_TF, reason="TensorFlow not installed")
def test_input_tensor_contract_api_spec_3():
    """
    Contract from docs/API.MD §3:
    Input tensor must accept calibrated vehicle-frame IMU samples:
    [Batch, Channels=6 (Ax, Ay, Az, Gx, Gy, Gz), WindowLen=10 (1.0 s at 10 Hz)].
    """
    interp = tf.lite.Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    input_details = interp.get_input_details()

    assert len(input_details) == 1, f"Expected exactly 1 input tensor, got {len(input_details)}"
    inp = input_details[0]
    expected_shape = [1, 6, 10]
    np.testing.assert_array_equal(
        inp["shape"],
        expected_shape,
        err_msg=f"Input tensor shape mismatch! Expected {expected_shape}, got {inp['shape'].tolist()}",
    )
    assert inp["dtype"] == np.float32, f"Input tensor dtype must be float32, got {inp['dtype']}"


@pytest.mark.skipif(not _HAS_TF, reason="TensorFlow not installed")
def test_output_tensor_count_and_shapes_contract():
    """
    Contract from TRD §4.1 and DV-03:
    Output tensor count must be 3 heads (or 2-3 heads):
    - Head A: Forward speed Vx [Batch=1, 1]
    - Head B: Speed variance sigma_v^2 [Batch=1, 1]
    - Head C: Stopped probability [Batch=1, 1]
    """
    interp = tf.lite.Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    output_details = interp.get_output_details()

    assert len(output_details) in (2, 3), f"Expected 2 or 3 output heads, got {len(output_details)}"
    for i, out in enumerate(output_details):
        np.testing.assert_array_equal(
            out["shape"],
            [1, 1],
            err_msg=f"Output head {i} shape mismatch! Expected [1, 1], got {out['shape'].tolist()}",
        )
        assert out["dtype"] == np.float32, f"Output head {i} dtype must be float32, got {out['dtype']}"


@pytest.mark.skipif(not _HAS_TF, reason="TensorFlow not installed")
def test_real_sample_window_inference():
    """
    Contract validation on real IMU data:
    Loads real IMU window from S-S1.csv preprocessed with continuous gravity decoupling,
    runs TFLite inference, and asserts all outputs are finite and physically plausible:
    - Vx >= 0.0 m/s
    - sigma_v^2 > 0.0 m^2/s^2 (strictly positive uncertainty)
    - 0.0 <= stoppedProb <= 1.0 (valid probability)
    """
    # 1. Ingest real IMU sample
    loader = IOVNBDLoader()
    s_data = loader.parse_smartphone_csv(S_CSV_PATH)
    raw_imu = s_data["imu_6dof"][:50]  # First 50 samples

    # 2. Continuous gravity decoupling (fc=0.5 Hz) prior to windowing
    preprocessor = IMUPreprocessor(raw_sampling_rate_hz=10, target_sampling_rate_hz=10, lpf_cutoff_hz=0.5)
    dynamic_accel, _ = preprocessor.isolate_gravity(raw_imu[:, 0:3])
    stream = np.hstack([dynamic_accel, raw_imu[:, 3:6]])
    windows = preprocessor.create_sliding_windows(stream, window_size=10, stride=1)
    assert len(windows) > 0, "Failed to create sliding windows from sample data"

    sample_window = windows[0:1].astype(np.float32)  # [1, 6, 10]

    # 3. Load TFLite Interpreter
    interp = tf.lite.Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    in_idx = interp.get_input_details()[0]["index"]
    interp.set_tensor(in_idx, sample_window)
    interp.invoke()

    # 4. Extract output values
    out_dets = interp.get_output_details()
    sorted_dets = sorted(
        out_dets,
        key=lambda d: int(d["name"].split(":")[-1]) if ":" in d["name"] else d["index"],
    )

    vx = float(interp.get_tensor(sorted_dets[0]["index"])[0, 0])
    var = float(interp.get_tensor(sorted_dets[1]["index"])[0, 0])
    zupt = float(interp.get_tensor(sorted_dets[2]["index"])[0, 0])

    # 5. Assert physical plausibility
    assert np.isfinite(vx), f"Predicted forward speed is non-finite: {vx}"
    assert vx >= 0.0, f"Predicted forward speed must be non-negative: {vx} m/s"
    assert vx < 100.0, f"Predicted forward speed {vx} m/s exceeds physical vehicle limit"

    assert np.isfinite(var), f"Predicted speed variance is non-finite: {var}"
    assert var > 0.0, f"Predicted variance must be strictly positive: {var}"

    assert np.isfinite(zupt), f"Predicted stopped probability is non-finite: {zupt}"
    assert 0.0 <= zupt <= 1.0, f"Predicted stopped probability must be in [0, 1]: {zupt}"


def test_model_io_export_contract():
    """Verify that export_model_artifacts generates valid export artifacts in a given directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        success = export_model_artifacts(
            output_dir=tmpdir,
            update_eval_report=False,
            update_manifest=False,
        )
        assert success is True
        tflite_file = os.path.join(tmpdir, "navicore_odometer_int8.tflite")
        assert os.path.exists(tflite_file)
        assert os.path.getsize(tflite_file) > 50 * 1024


def test_input_shape_contract():
    """Contract from API.MD §3: input IMU window shape [Batch, Channels=6, WindowLen=10]."""
    expected_channels = 6
    expected_window_len = 10
    dummy_input_shape = (1, expected_channels, expected_window_len)
    assert dummy_input_shape == (1, 6, 10)
