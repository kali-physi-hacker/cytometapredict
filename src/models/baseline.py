from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor


@dataclass
class BaselineConfig:
    model_type: str = "random_forest"  # or "ridge"
    random_state: int = 42


def build_model(cfg: BaselineConfig) -> MultiOutputRegressor:
    if cfg.model_type == "random_forest":
        base = RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            n_jobs=-1,
            random_state=cfg.random_state,
        )
    elif cfg.model_type == "ridge":
        base = Ridge(alpha=1.0, random_state=cfg.random_state)
    else:
        raise ValueError(f"Unsupported model_type: {cfg.model_type}")

    return MultiOutputRegressor(base)

