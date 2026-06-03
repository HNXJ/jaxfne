"""Shared base state for SDR-family optimizers."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class BaseSDRState:
    """Base optimizer state shared by SDR, GSDR, and AGSDR."""
    step: int = 0
    best_loss: float = float("inf")
    best_param: Optional[Any] = None
    reset_counter: int = 0
    var_sup_ema: float = 0.0
    var_unsup_ema: float = 0.0
    ema_decay: float = 0.99
