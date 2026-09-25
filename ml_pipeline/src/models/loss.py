"""
=============================================================================
NAVICORE AI: MULTI-TASK LOSS MODULE
Smart India Hackathon 2026 | Problem Statement ID: 260168
Theme: Smart Vehicles | Team: @enigm@ (Team ID: 132834)
=============================================================================
Combines:
1. Gaussian Negative Log-Likelihood (NLL) Loss:
   Models heteroscedastic uncertainty on forward speed regression (Vx, sigma^2).
   L_NLL = 0.5 * ( (y - y_hat)^2 / sigma^2 + ln(sigma^2) )
   This forces sigma^2 to learn true state-dependent variance, not a fixed scalar.
2. Binary Cross-Entropy (BCE) Loss:
   Supervises the ZUPT / idle classifier head against ground-truth stationary state.
   Stationary ground truth is derived from CAN wheel speed ground truth:
   v_gt < 0.05 m/s (approx 0.18 km/h) -> Stopped (ZUPT = 1.0), else Moving (ZUPT = 0.0).
3. Configuration-Driven Loss Weighting:
   Weights (w_nll, w_bce) are dynamically resolved from training_config.yaml per
   docs/CODE-STYLE.md §4 (never hardcoded).
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """Loads training configuration YAML file per docs/CODE-STYLE.md §4."""
    path = Path(config_path).resolve() if config_path else find_training_config_path()
    if path.exists() and _HAS_YAML:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            if isinstance(cfg, dict):
                return cfg

    # Fallback default configuration schema
    return {
        "training": {
            "loss_weights": {
                "speed_nll": 1.0,
                "zupt_bce": 0.5,
                "smoothness": 0.1,
            },
            "stationary_threshold_mps": 0.05,
        }
    }


def derive_stationary_label(
    speed_mps: Union[torch.Tensor, float],
    stationary_threshold_mps: float = 0.05,
) -> torch.Tensor:
    """
    Derives binary zero-velocity (ZUPT / stationary) ground truth labels from
    vehicle velocity ground truth.

    PHYSICAL DERIVATION:
    The Coventry University IO-VNBD dataset records vehicle speed from high-resolution
    CAN-bus wheel speed encoders (V-*.csv, columns 'ws_fl', 'ws_fr', 'ws_rl', 'ws_rr').
    At zero vehicle velocity (traffic stops, red lights, parking), encoder quantization,
    road slope creep, and vibration produce sensor readings between 0.00 and 0.04 m/s.
    Therefore, any speed strictly below stationary_threshold_mps (0.05 m/s = 0.18 km/h)
    is physically stationary (ZUPT = 1.0), and any speed >= 0.05 m/s is moving (ZUPT = 0.0).

    :param speed_mps: Tensor of vehicle ground truth speeds in m/s
    :param stationary_threshold_mps: Speed threshold below which vehicle is stationary (default: 0.05 m/s)
    :return: Float tensor of shape matching speed_mps with values in {0.0, 1.0}
    """
    if not isinstance(speed_mps, torch.Tensor):
        speed_tensor = torch.tensor(speed_mps, dtype=torch.float32)
    else:
        speed_tensor = speed_mps

    return (torch.abs(speed_tensor) < stationary_threshold_mps).float()


class MultiTaskOdometerLoss(nn.Module):
    """
    Multi-Task Loss combining Gaussian NLL speed regression and ZUPT BCE classification.
    Fully configurable via training_config.yaml.
    """

    def __init__(
        self,
        weight_speed_nll: Optional[float] = None,
        weight_zupt_bce: Optional[float] = None,
        stationary_threshold_mps: Optional[float] = None,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        from_logits: bool = True,
        variance_eps: float = 1e-5,
    ):
        """
        :param weight_speed_nll: Loss multiplier for Gaussian NLL speed regression.
                                 If None, loaded from training_config.yaml.
        :param weight_zupt_bce: Loss multiplier for ZUPT binary cross-entropy.
                                If None, loaded from training_config.yaml.
        :param stationary_threshold_mps: Speed threshold in m/s for stationary derivation.
        :param config_path: Explicit path to training_config.yaml.
        :param config: Direct configuration dictionary.
        :param from_logits: If True, pred_zupt is interpreted as raw logits and
                            F.binary_cross_entropy_with_logits is used for numerical stability.
        :param variance_eps: Numerical stability floor for variance clamp.
        """
        super().__init__()

        if config is not None:
            self.cfg = config
        else:
            self.cfg = load_training_config(config_path)

        train_cfg = self.cfg.get("training", {})
        loss_weights = train_cfg.get("loss_weights", {})

        # Load configurable loss weights from config
        self.w_nll = (
            float(weight_speed_nll)
            if weight_speed_nll is not None
            else float(loss_weights.get("speed_nll", 1.0))
        )
        self.w_bce = (
            float(weight_zupt_bce)
            if weight_zupt_bce is not None
            else float(loss_weights.get("zupt_bce", 0.5))
        )
        self.pos_weight = float(loss_weights.get("zupt_pos_weight", 1.0))
        self.stationary_threshold = (
            float(stationary_threshold_mps)
            if stationary_threshold_mps is not None
            else float(train_cfg.get("stationary_threshold_mps", 0.05))
        )

        self.from_logits = from_logits
        self.variance_eps = variance_eps

    def forward(
        self,
        pred_vx: torch.Tensor,
        pred_variance: torch.Tensor,
        pred_zupt: torch.Tensor,
        target_vx: torch.Tensor,
        target_zupt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes weighted multi-task loss.

        :param pred_vx: [Batch, 1] predicted forward speed (m/s)
        :param pred_variance: [Batch, 1] predicted speed variance sigma^2 > 0
        :param pred_zupt: [Batch, 1] predicted ZUPT logit (if from_logits=True) or probability
        :param target_vx: [Batch, 1] ground truth forward speed (m/s)
        :param target_zupt: [Batch, 1] ground truth binary stationary label.
                            If None, automatically derived from target_vx < threshold.
        :return: (total_loss, loss_speed_nll, loss_zupt_bce)
        """
        # Ensure dimensions match [Batch, 1]
        if target_vx.dim() == 1:
            target_vx = target_vx.unsqueeze(1)
        if pred_vx.dim() == 1:
            pred_vx = pred_vx.unsqueeze(1)
        if pred_variance.dim() == 1:
            pred_variance = pred_variance.unsqueeze(1)
        if pred_zupt.dim() == 1:
            pred_zupt = pred_zupt.unsqueeze(1)

        # Derive target_zupt from ground-truth velocity if not provided
        if target_zupt is None:
            target_zupt = derive_stationary_label(target_vx, self.stationary_threshold)
        elif target_zupt.dim() == 1:
            target_zupt = target_zupt.unsqueeze(1)

        # ---------------------------------------------------------------------
        # 1. Gaussian Negative Log-Likelihood (Heteroscedastic Uncertainty)
        # ---------------------------------------------------------------------
        # Floor variance for numerical stability
        var_clamped = torch.clamp(pred_variance, min=self.variance_eps)
        squared_error = (target_vx - pred_vx) ** 2
        # NLL = 0.5 * ( (y - y_hat)^2 / sigma^2 + ln(sigma^2) )
        nll_elementwise = 0.5 * (squared_error / var_clamped + torch.log(var_clamped))
        loss_speed_nll = torch.mean(nll_elementwise)

        # ---------------------------------------------------------------------
        # 2. Binary Cross-Entropy (ZUPT / Stationary Classifier)
        # ---------------------------------------------------------------------
        if self.from_logits:
            pos_weight_tensor = (
                torch.tensor([self.pos_weight], device=pred_zupt.device)
                if self.pos_weight > 1.0
                else None
            )
            loss_zupt_bce = F.binary_cross_entropy_with_logits(
                pred_zupt, target_zupt, pos_weight=pos_weight_tensor
            )
        else:
            p_clamped = torch.clamp(pred_zupt, min=1e-7, max=1.0 - 1e-7)
            loss_zupt_bce = F.binary_cross_entropy(p_clamped, target_zupt)

        # ---------------------------------------------------------------------
        # 3. Configurable Multi-Task Total Loss
        # ---------------------------------------------------------------------
        total_loss = (self.w_nll * loss_speed_nll) + (self.w_bce * loss_zupt_bce)

        return total_loss, loss_speed_nll, loss_zupt_bce
