# research/model_training.py
# Model training, versioning, artifact storage.
# NEVER import this module in execution/ or broker/ — research only.
#
# Rules enforced here:
#   - Deterministic seeding on every train run
#   - SHA-256 artifact integrity on every save and load
#   - No model deploys without validation_report.json
#   - deployment_approved must be set manually (human sign-off required)

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_MODEL_CLASSES = {"lightgbm", "xgboost"}


@dataclass
class ModelConfig:
    version: str                    # e.g. "eurusd_5m_v001"
    pair: str                       # e.g. "EUR/USD"
    timeframe: str                  # e.g. "5m"
    model_class: str                # "lightgbm" or "xgboost"
    model_params: dict
    random_seed: int = 42
    train_end_date: str = ""        # ISO date, filled at training time


@dataclass
class ValidationReport:
    fold_results: list[dict]        # per-fold metrics
    aggregate: dict                 # mean/std across folds
    deployment_approved: bool = False   # must be set to True manually
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    notes: str = ""


def seed_everything(seed: int) -> None:
    """
    Seed all random number generators for reproducibility.
    Call at the start of every training run.
    LightGBM and XGBoost receive seeds via their own params — do not rely on global state.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Random seeds set", extra={"seed": seed})


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: ModelConfig,
) -> object:
    """
    Train a LightGBM or XGBoost classifier.
    Returns the fitted model object.
    Validation set is used for early stopping only — not for hyperparameter selection.
    """
    assert config.model_class in SUPPORTED_MODEL_CLASSES, \
        f"Unsupported model class: {config.model_class}. Use: {SUPPORTED_MODEL_CLASSES}"

    seed_everything(config.random_seed)

    if config.model_class == "lightgbm":
        return _train_lgbm(X_train, y_train, X_val, y_val, config)
    elif config.model_class == "xgboost":
        return _train_xgboost(X_train, y_train, X_val, y_val, config)


def _train_lgbm(X_train, y_train, X_val, y_val, config: ModelConfig):
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm required: pip install lightgbm")

    # Class weights: handle imbalanced labels without SMOTE
    class_counts = y_train.value_counts()
    n_total = len(y_train)
    class_weight = {cls: n_total / (len(class_counts) * count)
                    for cls, count in class_counts.items()}

    sample_weight = y_train.map(class_weight).values

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbosity": -1,
        "random_state": config.random_seed,
        "n_jobs": 1,                # deterministic: single thread
        **config.model_params,
    }

    # Map labels {-1, 0, 1} → {0, 1, 2} for LightGBM multiclass
    label_map = {-1: 0, 0: 1, 1: 2}
    y_train_mapped = y_train.map(label_map)
    y_val_mapped = y_val.map(label_map)

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train_mapped,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val_mapped)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    logger.info(
        "LightGBM model trained",
        extra={
            "best_iteration": model.best_iteration_,
            "n_features": X_train.shape[1],
            "train_samples": len(X_train),
        }
    )
    return model


def _train_xgboost(X_train, y_train, X_val, y_val, config: ModelConfig):
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost required: pip install xgboost")

    label_map = {-1: 0, 0: 1, 1: 2}
    y_train_mapped = y_train.map(label_map)
    y_val_mapped = y_val.map(label_map)

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "seed": config.random_seed,
        "nthread": 1,   # deterministic
        **config.model_params,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train_mapped,
        eval_set=[(X_val, y_val_mapped)],
        early_stopping_rounds=200,
        verbose=False,
    )
    return model


def predict_proba(model, X: pd.DataFrame, model_class: str) -> np.ndarray:
    """
    Return probability array of shape (n_samples, 3).
    Columns: [p_short, p_abstain, p_long] after remapping from {0,1,2} → {-1,0,1}.
    """
    raw_proba = model.predict_proba(X)
    # raw_proba columns are [class_0, class_1, class_2] = [SHORT, ABSTAIN, LONG]
    return raw_proba  # shape (n, 3): [p_short, p_abstain, p_long]


def compute_signal(
    proba: np.ndarray,
    min_confidence: float = 0.55,
    min_margin: float = 0.10,
) -> tuple[int, float]:
    """
    Convert model output probabilities to a trading signal.

    Returns: (signal: int in {-1, 0, 1}, confidence: float)

    ABSTAIN conditions:
      1. Max probability < min_confidence
      2. Abstain class has highest probability
      3. Margin between top two classes < min_margin (uncertain)
    """
    p_short, p_abstain, p_long = proba[0], proba[1], proba[2]

    sorted_proba = sorted([p_short, p_abstain, p_long], reverse=True)
    max_proba = sorted_proba[0]
    second_proba = sorted_proba[1]

    if max_proba < min_confidence:
        return 0, max_proba

    if p_abstain == max_proba:
        return 0, p_abstain

    if (max_proba - second_proba) < min_margin:
        return 0, max_proba

    if p_long == max_proba:
        return 1, p_long

    return -1, p_short


# ------------------------------------------------------------------
# Artifact management
# ------------------------------------------------------------------

def save_model(
    model,
    config: ModelConfig,
    validation_report: Optional[ValidationReport],
    base_path: str,
) -> tuple[str, str]:
    """
    Save model artifact with manifest.
    Returns (artifact_dir, sha256).

    Directory structure:
      base_path/
        {version}_{date}/
          model.pkl
          model.manifest.json
          experiment_config.json
          validation_report.json   (if provided)
    """
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    artifact_dir = Path(base_path) / f"{config.version}_{date_str}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    pkl_path = artifact_dir / "model.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f, protocol=5)

    sha256 = hashlib.sha256(pkl_path.read_bytes()).hexdigest()

    # Manifest
    manifest = {
        "sha256": sha256,
        "version": config.version,
        "model_class": config.model_class,
        "pair": config.pair,
        "timeframe": config.timeframe,
        "train_end_date": config.train_end_date,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "deployment_approved": validation_report.deployment_approved
            if validation_report else False,
    }
    with open(artifact_dir / "model.manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Experiment config
    with open(artifact_dir / "experiment_config.json", "w") as f:
        config_dict = asdict(config)
        json.dump(config_dict, f, indent=2)

    # Validation report
    if validation_report:
        with open(artifact_dir / "validation_report.json", "w") as f:
            json.dump(asdict(validation_report), f, indent=2)

    logger.info(
        "Model artifact saved",
        extra={
            "artifact_dir": str(artifact_dir),
            "sha256": sha256,
            "deployment_approved": manifest["deployment_approved"],
        }
    )
    return str(artifact_dir), sha256


def load_model(artifact_dir: str, require_approved: bool = True) -> tuple[object, dict]:
    """
    Load model with integrity check.

    If require_approved=True (default for live deployment):
      Raises RuntimeError if deployment_approved != True in manifest.

    Returns (model, manifest_dict).
    """
    p = Path(artifact_dir)
    pkl_path = p / "model.pkl"
    manifest_path = p / "model.manifest.json"
    report_path = p / "validation_report.json"

    if not pkl_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {pkl_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Model manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())

    # Integrity check
    actual_sha256 = hashlib.sha256(pkl_path.read_bytes()).hexdigest()
    if actual_sha256 != manifest["sha256"]:
        raise RuntimeError(
            f"MODEL INTEGRITY CHECK FAILED. "
            f"Expected {manifest['sha256']}, got {actual_sha256}. "
            f"Artifact is corrupted or tampered. Do not proceed."
        )

    # Deployment approval check
    if require_approved:
        if not manifest.get("deployment_approved", False):
            raise RuntimeError(
                f"Model {manifest['version']} has not been approved for deployment. "
                f"Set deployment_approved=true in validation_report.json after human review."
            )
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if not report.get("deployment_approved", False):
                raise RuntimeError(
                    f"validation_report.json shows deployment_approved=false. "
                    f"Human sign-off required."
                )

    with open(pkl_path, "rb") as f:
        model = pickle.load(f)

    logger.info(
        "Model loaded",
        extra={
            "artifact_dir": artifact_dir,
            "sha256": actual_sha256,
            "version": manifest["version"],
            "deployment_approved": manifest.get("deployment_approved"),
        }
    )
    return model, manifest
