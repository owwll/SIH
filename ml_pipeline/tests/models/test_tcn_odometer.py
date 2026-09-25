"""
=============================================================================
NAVICORE AI: UNIT & CONTRACT TESTS FOR 1D-TCN VIRTUAL ODOMETER ARCHITECTURE
Smart India Hackathon 2026 | Problem Statement ID: 260168
=============================================================================
Tests:
1. Model forward pass on dummy batch (batch, 6, window_len) -> 3 output heads.
2. Output tensor shapes for Head A (Vx), Head B (sigma^2), and Head C (ZUPT logit).
3. Window length matches the configured resampling target from DV-01 (10 samples @ 10 Hz).
4. Temporal window length invariance via Global Average Pooling.
5. Parameter count aligns with mobile embedded budget (~122k parameters, < 500 KB).
"""

import os
import sys
import pytest
from pathlib import Path

# Add ml_pipeline and ml_pipeline/src to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from src.dataset.iovnbd_loader import (
    get_configured_training_rate_hz,
    get_configured_window_len_samples,
    load_training_config,
)


@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed in environment")
def test_tcn_odometer_3_heads_and_shapes():
    """
    Contract test: Input shape [Batch, 6 channels, window_len] ->
    Three outputs: Head A (Vx [Batch, 1]), Head B (sigma_v^2 [Batch, 1]), Head C (ZUPT [Batch, 1]).
    """
    from src.models.tcn_odometer import VirtualOdometerTCN

    batch_size = 4
    in_channels = 6
    window_len = 10

    model = VirtualOdometerTCN(in_channels=in_channels, stem_filters=32)
    model.eval()

    dummy_input = torch.randn(batch_size, in_channels, window_len)

    # 1. Forward with raw logit for Head C (default training mode)
    with torch.no_grad():
        vx, var, zupt_logit = model(dummy_input, return_prob=False)

    # Check shapes
    assert vx.shape == (batch_size, 1), f"Expected vx shape {(batch_size, 1)}, got {vx.shape}"
    assert var.shape == (batch_size, 1), f"Expected var shape {(batch_size, 1)}, got {var.shape}"
    assert zupt_logit.shape == (batch_size, 1), f"Expected zupt shape {(batch_size, 1)}, got {zupt_logit.shape}"

    # Head B: speed variance must be strictly positive (via Softplus)
    assert torch.all(var > 0), "Speed variance sigma^2 must be strictly positive"

    # 2. Forward with return_prob=True (inference mode)
    with torch.no_grad():
        _, _, zupt_prob = model(dummy_input, return_prob=True)

    assert torch.all((zupt_prob >= 0.0) & (zupt_prob <= 1.0)), (
        "ZUPT probability must be bounded in [0.0, 1.0]"
    )


@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed in environment")
def test_window_len_matches_configured_resampling_target():
    """
    Verifies that the model forward pass accepts an input whose window length
    matches the configured resampling target from DV-01 (10 samples @ 10 Hz = 1.0 s).
    """
    from src.models.tcn_odometer import VirtualOdometerTCN

    config = load_training_config()
    configured_rate = get_configured_training_rate_hz()
    configured_window_len = get_configured_window_len_samples()

    # DV-01 determined rate is 10 Hz and window_len is 10 samples (1.0 s)
    assert configured_rate == 10, f"Expected 10 Hz training rate, got {configured_rate}"
    assert configured_window_len == 10, f"Expected 10 sample window len, got {configured_window_len}"

    model = VirtualOdometerTCN(in_channels=6, stem_filters=32)
    model.eval()

    batch_x = torch.randn(2, 6, configured_window_len)
    with torch.no_grad():
        vx, var, zupt = model(batch_x)

    assert vx.shape == (2, 1)
    assert var.shape == (2, 1)
    assert zupt.shape == (2, 1)


@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed in environment")
def test_temporal_length_invariance():
    """
    Verifies that AdaptiveAvgPool1d allows the model to process any window length
    (e.g., 10 samples at 10 Hz, 25 samples, or 100 samples at 100 Hz).
    """
    from src.models.tcn_odometer import VirtualOdometerTCN

    model = VirtualOdometerTCN(in_channels=6, stem_filters=32)
    model.eval()

    for w_len in [10, 25, 50, 100]:
        x = torch.randn(1, 6, w_len)
        with torch.no_grad():
            vx, var, zupt = model(x)
        assert vx.shape == (1, 1)
        assert var.shape == (1, 1)
        assert zupt.shape == (1, 1)


@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed in environment")
def test_model_parameter_count_budget():
    """
    Asserts model parameter count aligns with TRD.md §4.2 and AI_ARCHITECTURE.md §3.1:
    Total parameters ~ 122,253 (~489 KB Float32, ~125 KB INT8).
    """
    from src.models.tcn_odometer import VirtualOdometerTCN

    model = VirtualOdometerTCN(in_channels=6, stem_filters=32)
    total_params = sum(p.numel() for p in model.parameters())

    # Range around intended architecture: 110k - 135k
    assert 110_000 <= total_params <= 135_000, (
        f"Expected ~122k parameters, got {total_params:,}"
    )
