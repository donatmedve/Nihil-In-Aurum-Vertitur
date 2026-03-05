# scripts/train_model.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
from features.pipeline import FeaturePipeline
from research.labels import get_label_distribution
from research.walk_forward import generate_folds, WalkForwardConfig, slice_fold
from research.model_training import (
    ModelConfig, ValidationReport, train_model,
    predict_proba, compute_signal, save_model, seed_everything
)

SYMBOL       = "eurusd"
VERSION      = "eurusd_5m_v001"
PIPELINE_DIR = f"artifacts/pipeline_{VERSION}"

# ── Evaluation thresholds ─────────────────────────────────────────────────
# FIX: lowered from 0.55/0.10 — those values are calibrated for binary models.
# With 3-class softmax, max probability rarely exceeds 0.55 even on strong signals.
# 0.40 confidence + 0.05 margin is the right starting point; tune up once model
# is confirmed to be learning (hit rate > 52%).
EVAL_MIN_CONFIDENCE = 0.40
EVAL_MIN_MARGIN     = 0.05


def evaluate_fold(proba, y_test):
    signals = []
    for i in range(len(proba)):
        sig, conf = compute_signal(
            proba[i],
            min_confidence=EVAL_MIN_CONFIDENCE,
            min_margin=EVAL_MIN_MARGIN,
        )
        signals.append((sig, conf))

    n_total   = len(signals)
    n_abstain = sum(1 for s, _ in signals if s == 0)
    correct   = 0
    traded    = 0
    for (sig, _), true in zip(signals, y_test.values):
        if sig != 0 and true != 0:
            traded += 1
            if sig == true:
                correct += 1

    hit_rate = correct / traded if traded > 0 else 0.0
    return {
        "n_bars":      n_total,
        "n_abstain":   n_abstain,
        "abstain_pct": round(n_abstain / n_total * 100, 1) if n_total > 0 else 0,
        "hit_rate":    round(hit_rate, 3),
        "n_traded":    traded,
    }


def main():
    seed_everything(42)

    print("Loading bars and labels...")
    with open(f"data/{SYMBOL}_5m_bars.pkl", "rb") as f:
        df = pickle.load(f)
    with open(f"data/{SYMBOL}_5m_labels.pkl", "rb") as f:
        labels = pickle.load(f)

    print(f"  Bars: {len(df)}, Labels: {len(labels)}")

    # ── Label distribution check ──────────────────────────────────────────
    # Healthy: LONG 20-30%, SHORT 20-30%, ABSTAIN 40-60%
    # If ABSTAIN > 70%: loosen LabelConfig (min_atr_pips / max_spread_pips)
    # If LONG or SHORT < 15%: labels are too sparse for the model to learn
    dist = get_label_distribution(labels)
    print(f"\nLabel distribution:")
    print(f"  LONG    : {dist.get('long_pct', '?')}  ({dist.get('long', 0):,} bars)")
    print(f"  SHORT   : {dist.get('short_pct', '?')}  ({dist.get('short', 0):,} bars)")
    print(f"  ABSTAIN : {dist.get('abstain_pct', '?')}  ({dist.get('abstain', 0):,} bars)")
    print(f"  Balanced: {dist.get('is_balanced', False)}")
    if not dist.get('is_balanced', False):
        print("  WARNING: Label imbalance detected. Check build_labels.py config.")

    print("\nTransforming features...")
    pipeline = FeaturePipeline.load(PIPELINE_DIR)
    features = pipeline.transform(df)
    labels   = labels.reindex(features.index)
    print(f"  Feature matrix: {features.shape}")

    # ── Walk-forward config ───────────────────────────────────────────────
    wf_config = WalkForwardConfig(
        train_bars=300_000,   # ~4 years of 5m bars during trading hours
        purge_bars=12,
        val_bars=75_000,      # ~1 year
        embargo_bars=12,
        test_bars=37_500,     # ~6 months
        step_bars=37_500,
        min_folds=3,
    )

    try:
        folds = generate_folds(len(features), wf_config)
    except ValueError as e:
        print(f"\nERROR: {e}")
        return

    print(f"\nGenerated {len(folds)} walk-forward folds")

    # ── Model config ──────────────────────────────────────────────────────
    # FIX: tuned params for stability across folds with varying class distributions:
    #   - learning_rate 0.02 → 0.05: faster convergence, less risk of early stopping
    #     killing the model after only a handful of rounds on hard folds
    #   - num_leaves 127 → 63: reduces overfitting on smaller folds
    #   - max_depth 8 → 6: same reason
    #   - min_child_samples 30 → 50: more regularization, smoother decision boundaries
    model_config = ModelConfig(
        version=VERSION,
        pair="EUR/USD",
        timeframe="5m",
        model_class="lightgbm",
        model_params={
            "n_estimators":      1000,
            "learning_rate":     0.05,
            "num_leaves":        63,
            "max_depth":         6,
            "min_child_samples": 50,
            "subsample":         0.8,
            "colsample_bytree":  0.8,
            # "class_weight": "balanced",  ← removed: _train_lgbm already applies
            #                                sample_weight manually; double-weighting
            #                                causes the model to over-trade.
        },
    )

    fold_results = []
    final_model  = None

    for fold in folds:
        print(f"\nFold {fold.fold_id}/{len(folds)}...")
        X_train, y_train, X_val, y_val, X_test, y_test = slice_fold(features, labels, fold)
        print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

        if len(X_train) < 100 or len(X_val) < 10:
            print("  Skipping — not enough samples")
            continue

        # ── Per-fold label distribution (helps diagnose bad folds) ───────
        fold_dist = get_label_distribution(y_train)
        print(f"  Train labels — L:{fold_dist.get('long_pct','?')}  "
              f"S:{fold_dist.get('short_pct','?')}  "
              f"A:{fold_dist.get('abstain_pct','?')}")

        model = train_model(X_train, y_train, X_val, y_val, model_config)

        # ── Early stopping diagnostic ─────────────────────────────────────
        if hasattr(model, 'best_iteration_') and model.best_iteration_ is not None:
            print(f"  Best iteration : {model.best_iteration_}  "
                  f"(low = early stopping fired too soon)")

        proba  = predict_proba(model, X_test, model_config.model_class)
        result = evaluate_fold(proba, y_test)
        result["fold_id"] = fold.fold_id
        fold_results.append(result)
        final_model = model

        print(f"  Hit rate: {result['hit_rate']:.1%}  |  "
              f"Abstain: {result['abstain_pct']:.0f}%  |  "
              f"Traded: {result['n_traded']:,} bars")

    if final_model is None:
        print("\nERROR: No folds completed.")
        return

    # ── Summary ───────────────────────────────────────────────────────────
    avg_hit     = np.mean([r["hit_rate"]    for r in fold_results])
    avg_abstain = np.mean([r["abstain_pct"] for r in fold_results])
    avg_traded  = np.mean([r["n_traded"]    for r in fold_results])

    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Folds completed : {len(fold_results)}")
    print(f"Avg hit rate    : {avg_hit:.1%}  (good if > 52%)")
    print(f"Avg abstain     : {avg_abstain:.0f}%  (good if 40-70%)")
    print(f"Avg trades/fold : {avg_traded:.0f}")
    print()

    if avg_abstain > 85:
        print("WARNING: Abstain rate still very high.")
        print("  → Check label distribution above. If ABSTAIN > 70% in labels,")
        print("    re-run build_labels.py with lower min_atr_pips (try 2.0) or")
        print("    higher max_spread_pips (try 6.0).")
    if avg_hit < 0.45 and avg_traded > 50:
        print("WARNING: Hit rate below 45%.")
        print("  → Model is learning something but not enough. Consider:")
        print("    1. More data (you have 10y which is good)")
        print("    2. Re-check pipeline.py for any remaining feature corruption")
        print("    3. Try a longer train window (increase train_bars)")

    report = ValidationReport(
        fold_results=fold_results,
        aggregate={
            "avg_hit_rate":    avg_hit,
            "avg_abstain_pct": avg_abstain,
            "avg_trades_fold": avg_traded,
            "n_folds":         len(fold_results),
            "eval_min_confidence": EVAL_MIN_CONFIDENCE,
            "eval_min_margin":     EVAL_MIN_MARGIN,
        },
        deployment_approved=False,
    )

    model_config.train_end_date = str(features.index[-1].date())
    artifact_dir, sha256 = save_model(
        final_model, model_config, report, base_path="artifacts/"
    )

    print(f"Model saved to : {artifact_dir}")
    print(f"SHA256         : {sha256}")
    print()
    print("NEXT: Review hit rate and abstain rate above, then if satisfied,")
    print("open these two files and set deployment_approved to true:")
    print(f"  {artifact_dir}\\validation_report.json")
    print(f"  {artifact_dir}\\model.manifest.json")


if __name__ == "__main__":
    main()