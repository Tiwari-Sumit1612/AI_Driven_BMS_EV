"""Parse NASA Li-ion Battery Aging .mat files (B0005, B0006, B0007, B0018)
into a tidy cycle-level feature table.

Usage:  python src/data_processing.py
Output: data/processed/cycle_features.csv
        data/processed/discharge_profiles.csv (downsampled curves for plots)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "FY08Q4"
OUT = ROOT / "data" / "processed"
CELLS = ["B0005", "B0006", "B0007", "B0018"]
Q_NOM = 2.0  # Ah, rated capacity


def _arr(x):
    return np.asarray(x).ravel().astype(float)


def load_cycles(cell):
    mat = loadmat(RAW / f"{cell}.mat", simplify_cells=True)
    return mat[cell]["cycle"]


def charge_features(d):
    t, v, i, temp = _arr(d["Time"]), _arr(d["Voltage_measured"]), _arr(d["Current_measured"]), _arr(d["Temperature_measured"])
    # constant-current phase ends when terminal voltage first reaches 4.2 V
    cc_idx = np.argmax(v >= 4.2) if np.any(v >= 4.2) else len(v) - 1
    return {
        "chg_cc_time_s": t[cc_idx] - t[0],
        "chg_total_time_s": t[-1] - t[0],
        "chg_time_to_tmax_s": t[np.argmax(temp)] - t[0],
        "chg_temp_max_C": temp.max(),
    }


def discharge_features(d):
    t, v, i, temp = _arr(d["Time"]), _arr(d["Voltage_measured"]), _arr(d["Current_measured"]), _arr(d["Temperature_measured"])
    on = np.abs(i) > 1.0  # load applied (2 A discharge)
    k = np.argmax(on)
    # equal-voltage-drop time: 3.9 V -> 3.6 V (a window every cell passes, independent of cut-off)
    def t_at(level):
        idx = np.where((v <= level) & on)[0]
        return t[idx[0]] if len(idx) else np.nan
    t500 = (t >= t[k]) & (t <= t[k] + 500)
    return {
        "dis_v_initial_V": v[max(k - 1, 0)],
        "dis_ir_drop_V": v[max(k - 1, 0)] - v[min(k + 1, len(v) - 1)],
        "dis_v_mean_first500s_V": v[t500].mean(),
        "dis_time_3p9_to_3p6_s": t_at(3.6) - t_at(3.9),
        "dis_temp_start_C": temp[k],
        "dis_temp_max_C": temp.max(),
        "dis_temp_rise_C": temp.max() - temp[k],
        "dis_time_to_tmax_s": t[np.argmax(temp)] - t[k],
    }


def process_cell(cell):
    cycles = load_cycles(cell)
    rows, profiles = [], []
    last_chg, last_re, last_rct = None, np.nan, np.nan
    n_dis = 0
    for c in cycles:
        typ, d = c["type"], c["data"]
        if typ == "charge":
            last_chg = charge_features(d)
        elif typ == "impedance":
            re, rct = float(np.real(d["Re"])), float(np.real(d["Rct"]))
            # a few EIS fits are clearly broken (negative / huge); skip them
            if 0 < re < 1:
                last_re = re
            if 0 < rct < 1:
                last_rct = rct
        elif typ == "discharge":
            n_dis += 1
            cap = float(d["Capacity"])
            row = {"cell": cell, "cycle": n_dis, "ambient_C": c["ambient_temperature"], "capacity_Ah": cap,
                   "soh_pct": cap / Q_NOM * 100, "eis_Re_ohm": last_re, "eis_Rct_ohm": last_rct}
            row.update(discharge_features(d))
            row.update(last_chg or {})
            rows.append(row)
            if n_dis in (1, 50, 100, 150) or n_dis == 168:
                t, v, temp = _arr(d["Time"]), _arr(d["Voltage_measured"]), _arr(d["Temperature_measured"])
                step = max(len(t) // 60, 1)
                for tt, vv, te in zip(t[::step], v[::step], temp[::step]):
                    profiles.append({"cell": cell, "cycle": n_dis, "time_s": tt, "voltage_V": vv, "temp_C": te})
    return pd.DataFrame(rows), pd.DataFrame(profiles)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    feats, profs = zip(*(process_cell(c) for c in CELLS))
    df, pf = pd.concat(feats, ignore_index=True), pd.concat(profs, ignore_index=True)
    raw_n = len(df)
    # cleaning: drop cycles with missing features or physically impossible values
    df = df[(df["capacity_Ah"] > 0.5) & (df["capacity_Ah"] < 2.2)]
    df[["eis_Re_ohm", "eis_Rct_ohm"]] = df.groupby("cell")[["eis_Re_ohm", "eis_Rct_ohm"]].transform(lambda s: s.ffill().bfill())
    df = df.dropna().reset_index(drop=True)
    df.to_csv(OUT / "cycle_features.csv", index=False)
    pf.to_csv(OUT / "discharge_profiles.csv", index=False)
    print(f"discharge cycles parsed: {raw_n}, kept after cleaning: {len(df)}")
    print(df.groupby("cell").agg(cycles=("cycle", "count"), cap_first=("capacity_Ah", "first"),
                                 cap_last=("capacity_Ah", "last"), soh_min=("soh_pct", "min")).round(3))


if __name__ == "__main__":
    main()
