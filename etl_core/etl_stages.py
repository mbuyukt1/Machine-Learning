import logging
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Tuple

logger = logging.getLogger("ETLStages")

class ProductionETLProcessor:
    """Production data engineering processor ensuring zero test set contamination."""
    def __init__(self, num_cols, cat_cols):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.state_ = {}

    def fit_training_stage(self, df: pd.DataFrame):
        medians = {}
        clips = {}
        for c in self.num_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            medians[c] = float(s.median())
            clips[c] = (float(s.quantile(0.01)), float(s.quantile(0.99)))

        modes = {}
        for c in self.cat_cols:
            s = df[c].astype(str).str.strip().replace("?", np.nan)
            m = s.dropna().mode()
            modes[c] = m.iloc[0] if not m.empty else "Missing"

        self.state_ = {
            "num_medians": medians,
            "num_clips": clips,
            "cat_modes": modes
        }
        return self

    def transform_stage(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.num_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(self.state_["num_medians"][c])
            lo, hi = self.state_["num_clips"][c]
            out[c] = out[c].clip(lower=lo, upper=hi)

        for c in self.cat_cols:
            out[c] = out[c].astype(str).str.strip().replace("?", np.nan).fillna(self.state_["cat_modes"][c])
        return out

    def engineer_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        res = df.copy()
        res["capital_net"] = res["capital-gain"] - res["capital-loss"]
        denom = np.maximum(res["age"] - 16.0, 1.0)
        res["hours_per_age"] = res["hours-per-week"] / denom
        return res
