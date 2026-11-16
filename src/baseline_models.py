"""Classical ML baselines for SOH estimation on NASA cells.

Protocols
  1. Leave-one-cell-out (LOCO): train on 3 cells, test on the unseen 4th.  <- main result
  2. Random 80/20 cycle split (shown only to quantify data leakage).
Feature sets
  full    : all 9 engineered features
  no_ewdt : without the equal-voltage-drop time (strongest single feature) - ablation

Usage:  python src/baseline_models.py
"""
from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "cycle_features.csv"
FIG = ROOT / "results" / "figures"
MET = ROOT / "results" / "metrics"
SEED = 42

FEATURES = ["eis_Re_ohm", "eis_Rct_ohm", "dis_ir_drop_V", "dis_v_mean_first500s_V", "dis_time_3p9_to_3p6_s",
            "dis_temp_rise_C", "dis_temp_max_C", "chg_cc_time_s", "chg_total_time_s"]
FEATURE_SETS = {"full": FEATURES, "no_ewdt": [f for f in FEATURES if f != "dis_time_3p9_to_3p6_s"]}
TARGET = "soh_pct"


def models():
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.8,
                                colsample_bytree=0.8, random_state=SEED),
        "SVR (RBF)": make_pipeline(StandardScaler(), SVR(C=30, epsilon=0.3, gamma="scale")),
    }


def score(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))), "R2": r2_score(y, p)}


def loco(df, feats):
    rows, preds = [], []
    for test_cell in sorted(df.cell.unique()):
        tr, te = df[df.cell != test_cell], df[df.cell == test_cell]
        for name, m in models().items():
            m.fit(tr[feats], tr[TARGET])
            p = m.predict(te[feats])
            rows.append({"model": name, "test_cell": test_cell, **score(te[TARGET], p)})
            preds.append(pd.DataFrame({"model": name, "cell": test_cell, "cycle": te.cycle.values,
                                       "soh_true": te[TARGET].values, "soh_pred": p}))
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def random_split(df, feats):
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED)
    out = []
    for name, m in models().items():
        m.fit(tr[feats], tr[TARGET])
        out.append({"model": name, **score(te[TARGET], m.predict(te[feats]))})
    return pd.DataFrame(out)


def figures(df, preds, prof):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"B0005": "#1f4e79", "B0006": "#c0504d", "B0007": "#4f8a3c", "B0018": "#7f6084"}

    # Fig 1: capacity fade
    fig, ax = plt.subplots(figsize=(7, 4))
    for c, g in df.groupby("cell"):
        ax.plot(g.cycle, g.capacity_Ah, ".-", ms=3, lw=1, label=c, color=colors[c])
    ax.axhline(1.6, ls="--", color="grey"); ax.text(2, 1.61, "EOL = 80% of 2.0 Ah (1.6 Ah)", color="grey", fontsize=9)
    ax.axhline(1.4, ls=":", color="grey"); ax.text(2, 1.41, "NASA test stop = 70% (1.4 Ah)", color="grey", fontsize=9)
    ax.set_xlabel("Discharge cycle"); ax.set_ylabel("Capacity (Ah)"); ax.set_title("Capacity fade of the four NASA cells")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / "fig1_capacity_fade.png", dpi=200); plt.close(fig)

    # Fig 2: discharge voltage & temperature profiles for B0005 at several ages
    p5 = prof[prof.cell == "B0005"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    cmap = plt.get_cmap("Blues")
    cyc = sorted(p5.cycle.unique())
    for j, k in enumerate(cyc):
        g = p5[p5.cycle == k]; col = cmap(0.35 + 0.65 * j / max(len(cyc) - 1, 1))
        a1.plot(g.time_s / 60, g.voltage_V, color=col, label=f"cycle {k}")
        a2.plot(g.time_s / 60, g.temp_C, color=col, label=f"cycle {k}")
    a1.set_xlabel("Time (min)"); a1.set_ylabel("Terminal voltage (V)"); a1.set_title("B0005 discharge voltage (2 A)")
    a2.set_xlabel("Time (min)"); a2.set_ylabel("Cell temperature (°C)"); a2.set_title("B0005 discharge temperature")
    a1.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG / "fig2_discharge_profiles.png", dpi=200); plt.close(fig)

    # Fig 3: health indicators vs cycle
    hi = [("eis_Re_ohm", "Electrolyte resistance Re (Ω)"), ("dis_ir_drop_V", "Voltage drop at load step (V)"),
          ("dis_temp_rise_C", "Temperature rise in discharge (°C)"), ("dis_time_3p9_to_3p6_s", "Time 3.9→3.6 V (s)")]
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.5))
    for ax, (f, lab) in zip(axs.ravel(), hi):
        for c, g in df.groupby("cell"):
            ax.plot(g.cycle, g[f], ".", ms=2.5, color=colors[c], label=c)
        ax.set_title(lab, fontsize=10); ax.set_xlabel("Cycle")
    axs[0, 0].legend(frameon=False, fontsize=8, markerscale=3); fig.tight_layout()
    fig.savefig(FIG / "fig3_health_indicators.png", dpi=200); plt.close(fig)

    # Fig 4: correlation heatmap
    cols = FEATURES + [TARGET]
    corr = df[cols].corr()
    short = ["Re", "Rct", "IR drop", "V mean 500s", "t 3.9→3.6V", "T rise", "T max", "CC chg time", "Chg time", "SOH"]
    fig, ax = plt.subplots(figsize=(7.5, 6.3))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(short)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
    fig.colorbar(im, fraction=0.046); ax.set_title("Pearson correlation of features and SOH")
    fig.tight_layout(); fig.savefig(FIG / "fig4_correlation.png", dpi=200); plt.close(fig)

    # Fig 5: predicted vs true SOH (best model, LOCO)
    best = preds[preds.model == "XGBoost"]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    for ax, (c, g) in zip(axs, best.groupby("cell")):
        ax.plot(g.cycle, g.soh_true, color="black", lw=1.2, label="Measured")
        ax.plot(g.cycle, g.soh_pred, color=colors[c], lw=1, label="XGBoost (cell unseen)")
        ax.set_title(f"Test cell {c}"); ax.set_xlabel("Cycle")
    axs[0].set_ylabel("SOH (%)"); axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig5_pred_vs_true.png", dpi=200); plt.close(fig)


def main():
    MET.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    prof = pd.read_csv(ROOT / "data" / "processed" / "discharge_profiles.csv")
    summary = {}
    for fs, feats in FEATURE_SETS.items():
        per_cell, preds = loco(df, feats)
        per_cell.to_csv(MET / f"loco_per_cell_{fs}.csv", index=False)
        agg = per_cell.groupby("model")[["MAE", "RMSE", "R2"]].agg(["mean", "std"]).round(3)
        agg.to_csv(MET / f"loco_summary_{fs}.csv")
        summary[fs] = per_cell
        print(f"\n=== LOCO, feature set: {fs} ===\n{agg}")
        print(per_cell.pivot(index="model", columns="test_cell", values="MAE").round(2))
        if fs == "full":
            figures(df, preds, prof)
            preds.to_csv(MET / "loco_predictions_full.csv", index=False)
    rs = random_split(df, FEATURES).round(3)
    rs.to_csv(MET / "random_split_full.csv", index=False)
    print(f"\n=== Random 80/20 split (leaky) ===\n{rs}")
    # feature importance from XGBoost trained on all cells
    xgb = models()["XGBoost"].fit(df[FEATURES], df[TARGET])
    imp = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values(ascending=False).round(3)
    imp.to_csv(MET / "xgb_feature_importance.csv")
    print(f"\n=== XGBoost feature importance ===\n{imp}")


if __name__ == "__main__":
    main()
