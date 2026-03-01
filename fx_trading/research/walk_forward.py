# research/walk_forward.py
# Rolling walk-forward framework for time-series model validation.
# Uses FIXED-SIZE training windows (rolling, not expanding).
# Rationale: FX regimes shift. Stale regime data dilutes recent signal.
#
# Window layout per fold:
#   [TRAIN][PURGE][VAL][EMBARGO][TEST]
#
# PURGE:   bars between train end and val start.
#          Prevents label leakage at boundaries (labels look forward max_holding_bars).
# EMBARGO: bars between val end and test start.
#          Prevents feature leakage (rolling windows near val boundary
#          would otherwise contaminate test features).

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    train_bars: int = 105_120       # ~12 months of 5m bars (trading hours)
    purge_bars: int = 12            # must equal max_holding_bars in LabelConfig
    val_bars: int = 17_520          # ~2 months
    embargo_bars: int = 12          # same as purge_bars
    test_bars: int = 8_760          # ~1 month per fold
    step_bars: int = 8_760          # advance by 1 test period per fold
    min_folds: int = 6              # abort if fewer folds possible


@dataclass
class WalkForwardFold:
    fold_id: int
    train_start: int
    train_end: int      # exclusive
    val_start: int
    val_end: int        # exclusive
    test_start: int
    test_end: int       # exclusive

    def describe(self, index: pd.DatetimeIndex) -> dict:
        """Return human-readable date boundaries for this fold."""
        def safe_date(i):
            if i < len(index):
                return str(index[i].date())
            return "out_of_bounds"
        return {
            "fold_id": self.fold_id,
            "train": f"{safe_date(self.train_start)} → {safe_date(self.train_end - 1)}",
            "val":   f"{safe_date(self.val_start)} → {safe_date(self.val_end - 1)}",
            "test":  f"{safe_date(self.test_start)} → {safe_date(self.test_end - 1)}",
            "train_bars": self.train_end - self.train_start,
            "val_bars":   self.val_end - self.val_start,
            "test_bars":  self.test_end - self.test_start,
        }


def generate_folds(
    n_bars: int,
    config: WalkForwardConfig,
) -> list[WalkForwardFold]:
    """
    Generate walk-forward folds. Validates no overlap between splits.
    Raises ValueError if minimum folds cannot be achieved.

    Example with defaults (n_bars = 262_800, ~3 years of 5m bars):
      Fold 0: train[0:105120] purge val[105132:122652] embargo test[122664:131424]
      Fold 1: train[8760:113880] ...
      ...
    """
    folds = []
    fold_id = 0
    train_start = 0

    while True:
        train_end   = train_start + config.train_bars
        val_start   = train_end + config.purge_bars
        val_end     = val_start + config.val_bars
        test_start  = val_end + config.embargo_bars
        test_end    = test_start + config.test_bars

        if test_end > n_bars:
            break

        fold = WalkForwardFold(
            fold_id=fold_id,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
        )

        _validate_fold(fold, n_bars, config)
        folds.append(fold)

        fold_id += 1
        train_start += config.step_bars

    if len(folds) < config.min_folds:
        raise ValueError(
            f"Only {len(folds)} folds possible with {n_bars} bars and current config. "
            f"Minimum required: {config.min_folds}. "
            f"Either reduce train/val/test window sizes or acquire more data."
        )

    logger.info(
        "Walk-forward folds generated",
        extra={"n_folds": len(folds), "n_bars": n_bars}
    )
    return folds


def _validate_fold(fold: WalkForwardFold, n_bars: int, config: WalkForwardConfig) -> None:
    """Assert all split boundaries are correct. Fatal if violated."""
    assert fold.train_end <= fold.val_start - config.purge_bars + config.purge_bars, \
        f"Fold {fold.fold_id}: train overlaps purge zone"
    assert fold.val_start > fold.train_end, \
        f"Fold {fold.fold_id}: val_start <= train_end"
    assert fold.test_start > fold.val_end, \
        f"Fold {fold.fold_id}: test_start <= val_end"
    assert fold.test_end <= n_bars, \
        f"Fold {fold.fold_id}: test_end {fold.test_end} > n_bars {n_bars}"
    assert fold.val_start - fold.train_end >= config.purge_bars, \
        f"Fold {fold.fold_id}: purge zone too small"
    assert fold.test_start - fold.val_end >= config.embargo_bars, \
        f"Fold {fold.fold_id}: embargo zone too small"


def slice_fold(
    df: pd.DataFrame,
    labels: pd.Series,
    fold: WalkForwardFold,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Slice bar DataFrame and labels into (train, val, test) splits for a fold.
    Drops NaN labels from each split.
    Returns (X_train, y_train, X_val, y_val, X_test, y_test).

    Note: df here should be the feature matrix (output of pipeline.transform()),
    not the raw bar DataFrame.
    """
    def _clean(features, lbl):
        valid = lbl.dropna()
        features_aligned = features.loc[features.index.isin(valid.index)]
        lbl_aligned = valid.loc[valid.index.isin(features_aligned.index)]
        return features_aligned, lbl_aligned.astype(int)

    train_feat = df.iloc[fold.train_start: fold.train_end]
    train_lbl  = labels.iloc[fold.train_start: fold.train_end]

    val_feat   = df.iloc[fold.val_start: fold.val_end]
    val_lbl    = labels.iloc[fold.val_start: fold.val_end]

    test_feat  = df.iloc[fold.test_start: fold.test_end]
    test_lbl   = labels.iloc[fold.test_start: fold.test_end]

    X_train, y_train = _clean(train_feat, train_lbl)
    X_val,   y_val   = _clean(val_feat,   val_lbl)
    X_test,  y_test  = _clean(test_feat,  test_lbl)

    logger.info(
        "Fold sliced",
        extra={
            "fold_id": fold.fold_id,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
        }
    )

    return X_train, y_train, X_val, y_val, X_test, y_test
