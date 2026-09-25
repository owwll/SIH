"""
=============================================================================
NAVICORE AI: UNIT TESTS FOR MULTI-TASK ODOMETER LOSS FUNCTION
Smart India Hackathon 2026 | Problem Statement ID: 260168
=============================================================================
Tests:
1. Multi-task loss is finite and differentiable with non-zero backward gradients.
2. Loss strictly decreases monotonically as predictions approach ground truth.
3. Gaussian NLL heteroscedastic uncertainty dynamics (sigma^2 learns variance).
4. Stationary (ZUPT) label derivation from ground truth velocity threshold (< 0.05 m/s).
5. Configuration-driven loss weighting (training_config.yaml).
"""

import sys
import pytest
from pathlib import Path
import torch

# Add ml_pipeline and ml_pipeline/src to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

from src.models.loss import MultiTaskOdometerLoss, derive_stationary_label, load_training_config


def test_loss_finite_and_differentiable():
    """Verifies that multi-task loss is finite and differentiable with respect to all 3 heads."""
    loss_fn = MultiTaskOdometerLoss(from_logits=True)

    batch_size = 4
    pred_vx = torch.randn(batch_size, 1, requires_grad=True)
    pred_var = torch.exp(torch.randn(batch_size, 1))  # strictly positive
    pred_var.requires_grad_(True)
    pred_zupt = torch.randn(batch_size, 1, requires_grad=True)

    target_vx = torch.tensor([[10.0], [0.0], [5.5], [18.2]])
    target_zupt = torch.tensor([[0.0], [1.0], [0.0], [0.0]])

    total_loss, l_speed, l_zupt = loss_fn(
        pred_vx, pred_var, pred_zupt, target_vx, target_zupt
    )

    # 1. Finite check
    assert torch.isfinite(total_loss), "Total loss must be finite"
    assert torch.isfinite(l_speed), "Speed NLL loss must be finite"
    assert torch.isfinite(l_zupt), "ZUPT BCE loss must be finite"

    # 2. Differentiable check
    total_loss.backward()

    assert pred_vx.grad is not None and torch.all(torch.isfinite(pred_vx.grad))
    assert pred_var.grad is not None and torch.all(torch.isfinite(pred_var.grad))
    assert pred_zupt.grad is not None and torch.all(torch.isfinite(pred_zupt.grad))


def test_loss_decreases_as_predictions_approach_ground_truth():
    """Verifies that loss decreases when predictions move closer to ground truth."""
    loss_fn = MultiTaskOdometerLoss(from_logits=True)

    target_vx = torch.tensor([[12.0]])
    target_zupt = torch.tensor([[0.0]])  # Moving

    # Distant / inaccurate predictions
    pred_vx_bad = torch.tensor([[30.0]])
    pred_var_bad = torch.tensor([[1.0]])
    pred_zupt_bad = torch.tensor([[5.0]])  # Strongly predicts stopped (wrong)

    # Close / accurate predictions
    pred_vx_good = torch.tensor([[12.1]])
    pred_var_good = torch.tensor([[1.0]])
    pred_zupt_good = torch.tensor([[-5.0]])  # Strongly predicts moving (correct)

    loss_bad, l_speed_bad, l_zupt_bad = loss_fn(
        pred_vx_bad, pred_var_bad, pred_zupt_bad, target_vx, target_zupt
    )
    loss_good, l_speed_good, l_zupt_good = loss_fn(
        pred_vx_good, pred_var_good, pred_zupt_good, target_vx, target_zupt
    )

    assert loss_good < loss_bad, (
        f"Expected loss_good ({loss_good:.4f}) < loss_bad ({loss_bad:.4f})"
    )
    assert l_speed_good < l_speed_bad, "Speed NLL must decrease with closer velocity"
    assert l_zupt_good < l_zupt_bad, "ZUPT BCE must decrease with correct classification"


def test_learned_uncertainty_gradient():
    """
    Verifies that Gaussian NLL encourages sigma^2 to learn true residual magnitude:
    - If error^2 > sigma^2: dL/d(sigma^2) < 0 (pushes variance UP)
    - If error^2 < sigma^2: dL/d(sigma^2) > 0 (pushes variance DOWN)
    """
    loss_fn = MultiTaskOdometerLoss()

    target_vx = torch.tensor([[10.0]])
    target_zupt = torch.tensor([[0.0]])

    # Case 1: Large residual (error^2 = 25.0), small variance (sigma^2 = 1.0)
    pred_vx = torch.tensor([[5.0]])
    pred_var_small = torch.tensor([[1.0]], requires_grad=True)
    pred_zupt = torch.tensor([[-2.0]])

    total_loss_1, _, _ = loss_fn(pred_vx, pred_var_small, pred_zupt, target_vx, target_zupt)
    total_loss_1.backward()

    # Gradient should be negative: increasing sigma^2 will lower the loss
    assert pred_var_small.grad.item() < 0.0, "Expected negative gradient to expand variance"

    # Case 2: Zero residual (error^2 = 0.0), large variance (sigma^2 = 10.0)
    pred_vx_exact = torch.tensor([[10.0]])
    pred_var_large = torch.tensor([[10.0]], requires_grad=True)

    total_loss_2, _, _ = loss_fn(pred_vx_exact, pred_var_large, pred_zupt, target_vx, target_zupt)
    total_loss_2.backward()

    # Gradient should be positive: decreasing sigma^2 will lower the loss (penalizing over-uncertainty)
    assert pred_var_large.grad.item() > 0.0, "Expected positive gradient to contract variance"


def test_stationary_label_derivation():
    """
    Verifies that stationary label is derived from ground-truth velocity < 0.05 m/s.
    """
    # Test values in m/s
    speeds = torch.tensor([0.00, 0.02, 0.049, 0.051, 1.5, 20.0])
    labels = derive_stationary_label(speeds, stationary_threshold_mps=0.05)

    expected = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    assert torch.all(labels == expected), f"Expected {expected}, got {labels}"


def test_automatic_target_zupt_derivation():
    """Verifies that MultiTaskOdometerLoss derives target_zupt when passed target_zupt=None."""
    loss_fn = MultiTaskOdometerLoss()

    pred_vx = torch.tensor([[0.0], [10.0]])
    pred_var = torch.tensor([[1.0], [1.0]])
    pred_zupt = torch.tensor([[2.0], [-2.0]])
    target_vx = torch.tensor([[0.01], [10.0]])  # Sample 1: stopped (0.01 < 0.05), Sample 2: moving

    # Pass target_zupt=None
    total_loss, _, l_zupt = loss_fn(pred_vx, pred_var, pred_zupt, target_vx, target_zupt=None)
    assert torch.isfinite(total_loss)
    assert l_zupt < 0.5  # Correctly predicted stopped and moving, so BCE is small


def test_configuration_driven_loss_weights():
    """Verifies that loss weights are loaded from training_config.yaml and not hardcoded."""
    config = load_training_config()
    expected_nll_weight = config["training"]["loss_weights"]["speed_nll"]
    expected_bce_weight = config["training"]["loss_weights"]["zupt_bce"]

    loss_fn = MultiTaskOdometerLoss()
    assert loss_fn.w_nll == expected_nll_weight
    assert loss_fn.w_bce == expected_bce_weight
    assert loss_fn.w_nll == 1.0
    assert loss_fn.w_bce == 0.5
