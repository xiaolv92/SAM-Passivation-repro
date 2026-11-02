"""
Step 2 - six regression models + SHAP importance (reproduction of Su et al. Fig. 2).

Inputs : outputs/fingerprints_train.csv, outputs/features.json
Outputs: outputs/model_metrics.csv           (train/test RMSE & MAE, or LOOCV)
         outputs/model_metrics_cv.csv        (5-fold CV, or LOOCV)
         outputs/shap_ranking.csv            (mean |SHAP| over 5 seeds)
         outputs/model_rf.joblib             (used by step3)
         outputs/target.json                 (which target column was modelled)
         outputs/figures/fig2_parity.png, fig2_metrics.png, fig2_shap_beeswarm.png

Small-n policy
    n >= 30 : 15% hold-out split (random_state=31) + 5-fold CV, exactly as the paper
    8..29   : leave-one-out CV only (a 12-point test set is pure noise)
    n < 8   : refuse to fit; add data or run tests/make_demo_dataset.py

Usage:
    python src/step2_features_train.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FIG, OUT, load_features, mae, rmse, target_name  # noqa: E402

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 1.2,
                     "xtick.direction": "in", "ytick.direction": "in"})
warnings.filterwarnings("ignore", category=ConvergenceWarning)
SEED = 31


def get_models(n_features):
    """Hyperparameters from the paper's Table S2; GP length-scale vector follows n_features."""
    return {
        "RF": RandomForestRegressor(n_estimators=30, max_depth=6, random_state=SEED),
        "GB": GradientBoostingRegressor(n_estimators=35, max_depth=7, random_state=SEED),
        "SVR": SVR(C=5, epsilon=0.005),
        "MLP": MLPRegressor(hidden_layer_sizes=(36, 24, 8), alpha=0.015, activation="relu",
                            max_iter=10000, solver="adam", tol=1e-7, random_state=SEED),
        "GP": GaussianProcessRegressor(
            kernel=ConstantKernel(0.05, (4.5e-3, 1e-1))
            * Matern(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1.0), nu=0.5)
            + WhiteKernel(1e-4, (1e-8, 1e-2)),
            normalize_y=True, n_restarts_optimizer=10, random_state=SEED),
        "XGB": XGBRegressor(objective="reg:squarederror", n_estimators=80, learning_rate=0.10,
                            max_depth=6, min_child_weight=1, gamma=0, subsample=0.8,
                            colsample_bytree=0.8, reg_alpha=0.001, random_state=SEED),
    }


def parity(preds, y_tr, y_te, label):
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), dpi=200)
    for ax, (name, (p_tr, p_te)) in zip(axes.ravel(), preds.items()):
        hi = max(y_tr.max(), y_te.max()) * 1.05
        ax.plot([0, hi], [0, hi], ":", color="dimgray", lw=1.1)
        ax.scatter(y_tr, p_tr, s=30, c="cornflowerblue", edgecolors="mediumblue", alpha=.85,
                   label="Train")
        ax.scatter(y_te, p_te, s=30, c="salmon", edgecolors="tomato", alpha=.9, label="Test")
        ax.set_title(f"{name}  ({label} RMSE={rmse(y_te, p_te):.3f})", fontsize=9)
        ax.set_xlabel("True value", fontsize=8)
        ax.set_ylabel("Predicted", fontsize=8)
        ax.tick_params(labelsize=8)
    axes.ravel()[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_parity.png"))
    plt.close(fig)


def main():
    df = pd.read_csv(os.path.join(OUT, "fingerprints_train.csv"))
    features = load_features()
    X = df[features].astype(float)
    target = target_name(df)
    y = df[target].astype(float)
    n = len(X)

    print(f"[step2] n={n}  features={len(features)}  target={target}")
    if n < 8:
        # drop artefacts from a previous (larger) run so step3 cannot silently reuse them
        for stale in ("model_rf.joblib", "target.json", "model_metrics.csv",
                      "model_metrics_cv.csv", "shap_ranking.csv"):
            path = os.path.join(OUT, stale)
            if os.path.exists(path):
                os.remove(path)
        print("  n<8: not enough paired device data to fit any model.")
        print("  -> fill data/dataset.xlsx (schema in data/README.md) or run")
        print("     python tests\\make_demo_dataset.py  and then")
        print("     python src\\step1_dataset.py --dataset dataset_demo")
        return

    with open(os.path.join(OUT, "target.json"), "w", encoding="utf-8") as f:
        json.dump({"target": target, "n": n, "n_features": len(features)}, f, indent=2)

    models = get_models(len(features))
    if n < 25:
        models.pop("GP", None)          # degenerates to a constant at this sample size
        models.pop("MLP", None)

    metrics, preds, cv, loo_pred = [], {}, [], {}
    if n >= 30:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=SEED)
        split_label = "test"
    else:
        X_tr, X_te, y_tr, y_te = X, X, y, y
        split_label = "LOOCV"

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        p_tr = np.ravel(model.predict(X_tr))
        if n >= 30:
            p_te = np.ravel(model.predict(X_te))
        else:
            p_te = np.ravel(cross_val_predict(model, X, y, cv=LeaveOneOut()))
            loo_pred[name] = p_te
        preds[name] = (p_tr, p_te)
        metrics.append(dict(model=name,
                            rmse_train=round(rmse(y_tr, p_tr), 4),
                            rmse_test=round(rmse(y_te, p_te), 4),
                            mae_train=round(mae(y_tr, p_tr), 4),
                            mae_test=round(mae(y_te, p_te), 4)))
        print(f"  {name:4s} RMSE {metrics[-1]['rmse_train']:.3f}/{metrics[-1]['rmse_test']:.3f}"
              f"   MAE {metrics[-1]['mae_train']:.3f}/{metrics[-1]['mae_test']:.3f}")

    pd.DataFrame(metrics).to_csv(os.path.join(OUT, "model_metrics.csv"), index=False)

    # honest generalisation estimate: 5-fold for n>=30, LOOCV below that
    cv_split = KFold(5, shuffle=True, random_state=SEED) if n >= 30 else LeaveOneOut()
    for name, model in models.items():
        y_hat = loo_pred.get(name)
        if y_hat is None:
            y_hat = cross_val_predict(model, X, y, cv=cv_split)
        cv.append(dict(model=name, rmse=round(rmse(y, y_hat), 4),
                       mae=round(mae(y, y_hat), 4)))
        print(f"  CV {name:4s} RMSE={cv[-1]['rmse']:.3f}  MAE={cv[-1]['mae']:.3f}")
    pd.DataFrame(cv).to_csv(os.path.join(OUT, "model_metrics_cv.csv"), index=False)
    parity(preds, y_tr.values, y_te.values, split_label)

    # bar chart
    m = pd.DataFrame(metrics)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), dpi=200)
    for i, (col, lab) in enumerate([("rmse", "RMSE"), ("mae", "MAE")]):
        w, xs = 0.36, np.arange(len(m))
        ax[i].bar(xs - w / 2, m[f"{col}_train"], w, label="Train", color="cornflowerblue",
                  edgecolor="mediumblue")
        ax[i].bar(xs + w / 2, m[f"{col}_test"], w, label=f"Test/{split_label}",
                  color="salmon", edgecolor="tomato")
        ax[i].set_xticks(xs)
        ax[i].set_xticklabels(m["model"])
        ax[i].set_ylabel(lab)
        ax[i].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_metrics.png"))
    plt.close(fig)

    # ---- importance ---------------------------------------------------------------
    rf = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=SEED).fit(X, y)
    joblib.dump(rf, os.path.join(OUT, "model_rf.joblib"))
    if n >= 20:
        svs = [shap.TreeExplainer(
            RandomForestRegressor(n_estimators=30, max_depth=6, random_state=s).fit(X, y)
        ).shap_values(X) for s in (31, 41, 51, 61, 71)]
        std = np.std([np.abs(v).mean(axis=0) for v in svs], axis=0)
        sv = np.mean(svs, axis=0)
        imp = np.abs(sv).mean(axis=0)
        shap.summary_plot(sv, X, show=False, max_display=15)
        plt.gcf().set_size_inches(6.6, 5.0)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, "fig2_shap_beeswarm.png"), dpi=200, bbox_inches="tight")
        plt.close()
    else:
        perm = permutation_importance(rf, X, y, n_repeats=30, random_state=SEED)
        imp, std = perm.importances_mean, perm.importances_std

    rank = (pd.DataFrame({"feature": features, "importance": imp, "seed_std": std})
            .sort_values("importance", ascending=False).reset_index(drop=True))
    rank["rank"] = rank.index + 1
    rank.to_csv(os.path.join(OUT, "shap_ranking.csv"), index=False)
    print(rank.head(10).to_string(index=False))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
