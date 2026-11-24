"""Eksperimen P4: fitur lag dan rolling mean untuk peramalan energi harian.

Jalankan dari root repo (setelah audit_data.py menghasilkan data/daily_energy.csv):
    python src/run_experiments.py

Keputusan metodologi mengikuti docs/decisions.md:
  - target  : energy_kwh harian (NaN dibiarkan, tidak diimputasi)
  - split   : kronologis (train / validation / test), tanpa pengacakan
  - fitur   : A = kalender; B = A + lag1 + lag7; C = B + rolling mean 7 dan 14 hari
  - leakage : lag dan rolling selalu dihitung dari series yang digeser 1 hari
  - evaluasi: semua baseline dan skenario dinilai pada himpunan hari yang SAMA

Protokol evaluasi:
  1. validation       : fit di train, tuning hyperparameter di validation
  2. test_walkforward : expanding window, fit ulang tiap bulan (train+validation+bulan
                        test sebelumnya), hyperparameter tetap dari validation.
                        Test dihitung SEKALI, tidak dipakai untuk memilih model.
  3. random_split     : pembanding untuk H2 (BUKAN evaluasi valid untuk deret waktu)

Keluaran: results/*.csv, results/run_log.json, figures/*.png
"""
import json
import platform
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_RANDOM_SEEDS = 5
SPLITS = {
    "train": ("2006-12-16", "2008-12-31"),
    "validation": ("2009-01-01", "2009-12-31"),
    "test": ("2010-01-01", "2010-11-26"),
}
GRIDS = {
    "Ridge": [{"alpha": a} for a in (0.01, 0.1, 1, 10, 100)],
    "RandomForest": [
        {"max_depth": md, "min_samples_leaf": ml}
        for md in (5, 10, None)
        for ml in (1, 5)
    ],
}

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)


# ----------------------------------------------------------------- utilitas
def mae(a, p):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(p))))


def rmse(a, p):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(p)) ** 2)))


def make_model(name, params):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=params["alpha"]))
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=SEED,
        n_jobs=-1,
    )


# --------------------------------------------------------------------- data
d = pd.read_csv("data/daily_energy.csv", index_col="date", parse_dates=True)
d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))  # hari kontinu
y = d["energy_kwh"]

# --------------------------------------------------------------------- fitur
X = pd.DataFrame(index=y.index)
for k in range(1, 7):
    X[f"dow_{k}"] = (y.index.dayofweek == k).astype(float)
for m in range(2, 13):
    X[f"month_{m}"] = (y.index.month == m).astype(float)
CAL = list(X.columns)

shifted = y.shift(1)  # hanya informasi hari <= t-1
X["lag1"] = y.shift(1)
X["lag7"] = y.shift(7)
X["roll7"] = shifted.rolling(7).mean()  # NaN bila ada hari NaN di jendela
X["roll14"] = shifted.rolling(14).mean()
HIST = ["lag1", "lag7", "roll7", "roll14"]

SCEN = {
    "A": CAL,
    "B": CAL + ["lag1", "lag7"],
    "C": CAL + ["lag1", "lag7", "roll7", "roll14"],
}

# Himpunan hari evaluasi bersama: target valid DAN semua fitur historis valid
common = y.notna() & X[HIST].notna().all(axis=1)


def split_idx(name, only_common=True):
    s, e = SPLITS[name]
    idx = y.loc[s:e].index
    return idx[common.loc[idx].values] if only_common else idx


tr, va, te = split_idx("train"), split_idx("validation"), split_idx("test")

# ------------------------------------------- tabel karakteristik & jumlah hari
char_rows, count_rows = [], []
for name in SPLITS:
    idx = split_idx(name, only_common=False)
    v = y.loc[idx].dropna()
    ev = common.loc[idx]
    char_rows.append({
        "split": name, "start": idx.min().date(), "end": idx.max().date(),
        "n_days": len(idx), "n_nan_days": int(y.loc[idx].isna().sum()),
        "mean_kwh": round(v.mean(), 2), "std_kwh": round(v.std(), 2),
        "min_kwh": round(v.min(), 2), "max_kwh": round(v.max(), 2),
    })
    count_rows.append({
        "split": name, "n_days": len(idx),
        "y_nan": int(y.loc[idx].isna().sum()),
        "lost_due_to_features": int((y.loc[idx].notna() & ~ev).sum()),
        "n_evaluated": int(ev.sum()),
        "pct_evaluated": round(float(ev.mean()) * 100, 1),
    })
pd.DataFrame(char_rows).to_csv("results/dataset_table.csv", index=False)
pd.DataFrame(count_rows).to_csv("results/eval_counts.csv", index=False)
print("== Karakteristik dataset ==")
print(pd.DataFrame(char_rows).to_string(index=False))
print("\n== Jumlah hari dievaluasi (himpunan bersama) ==")
print(pd.DataFrame(count_rows).to_string(index=False))


def fit_predict(scen, model, params, tr_idx, te_idx):
    cols = SCEN[scen]
    m = make_model(model, params)
    m.fit(X.loc[tr_idx, cols], y.loc[tr_idx])
    return pd.Series(m.predict(X.loc[te_idx, cols]), index=te_idx)


def row(protocol, scen, model, idx, pred, params=""):
    return dict(protocol=protocol, scenario=scen, model=model, n_eval=len(idx),
                mae=mae(y.loc[idx], pred), rmse=rmse(y.loc[idx], pred),
                mae_std=np.nan, params=json.dumps(params) if params else "")


def baseline_rows(protocol, idx):
    return [
        row(protocol, "naive", "-", idx, X.loc[idx, "lag1"]),
        row(protocol, "seasonal_naive", "-", idx, X.loc[idx, "lag7"]),
    ]


# ---------------------------------------------- 1. validation + tuning (train)
best = {}
val_rows = baseline_rows("validation", va)
for scen in SCEN:
    for model, grid in GRIDS.items():
        scored = []
        for p in grid:
            pred = fit_predict(scen, model, p, tr, va)
            scored.append((mae(y.loc[va], pred), p))
        score, p_best = min(scored, key=lambda t: t[0])
        best[(scen, model)] = p_best
        pred = fit_predict(scen, model, p_best, tr, va)
        val_rows.append(row("validation", scen, model, va, pred, p_best))
print("\nTuning selesai.")

# ------------------------------------------ 2. test: walk-forward (sekali jalan)
test_rows = baseline_rows("test_walkforward", te)
test_pred = {}
months = te.to_period("M")
for scen in SCEN:
    for model in GRIDS:
        preds = []
        for per in months.unique():
            te_m = te[months == per]
            tr_m = common.index[common.values & (common.index < per.start_time)]
            preds.append(fit_predict(scen, model, best[(scen, model)], tr_m, te_m))
        pred = pd.concat(preds).loc[te]
        test_pred[(scen, model)] = pred
        test_rows.append(row("test_walkforward", scen, model, te, pred, best[(scen, model)]))
print("Walk-forward test selesai.")

# -------------------------------------- 3. random split (pembanding H2, bukan valid)
all_idx = common.index[common.values]
all_idx = all_idx[all_idx <= pd.Timestamp(SPLITS["test"][1])]
rand_raw = []
for seed in range(SEED, SEED + N_RANDOM_SEEDS):
    perm = np.random.default_rng(seed).permutation(len(all_idx))
    n_te = int(round(0.2 * len(all_idx)))
    te_r = all_idx[np.sort(perm[:n_te])]
    tr_r = all_idx[np.sort(perm[n_te:])]
    rand_raw += baseline_rows("random_split", te_r)
    for scen in SCEN:
        for model in GRIDS:
            pred = fit_predict(scen, model, best[(scen, model)], tr_r, te_r)
            rand_raw.append(row("random_split", scen, model, te_r, pred, best[(scen, model)]))
rr = pd.DataFrame(rand_raw)
rand_rows = []
for (scen, model), g in rr.groupby(["scenario", "model"], sort=False):
    rand_rows.append(dict(protocol="random_split", scenario=scen, model=model,
                          n_eval=int(g["n_eval"].iloc[0]), mae=g["mae"].mean(),
                          rmse=g["rmse"].mean(), mae_std=g["mae"].std(),
                          params=g["params"].iloc[0]))
print("Random split selesai.")

# ------------------------------------------------------------ gabung metrik
M = pd.DataFrame(val_rows + test_rows + rand_rows)
M["mae_vs_seasonal_naive_pct"] = np.nan
M["mae_vs_scenarioA_pct"] = np.nan
for proto, g in M.groupby("protocol"):
    base = g.loc[g.scenario == "seasonal_naive", "mae"].iloc[0]
    M.loc[g.index, "mae_vs_seasonal_naive_pct"] = (g["mae"] / base - 1) * 100
    for model in GRIDS:
        a = g[(g.scenario == "A") & (g.model == model)]["mae"].iloc[0]
        sel = g.index[g.model == model]
        M.loc[sel, "mae_vs_scenarioA_pct"] = (M.loc[sel, "mae"] / a - 1) * 100
M = M.round({"mae": 4, "rmse": 4, "mae_std": 4,
             "mae_vs_seasonal_naive_pct": 2, "mae_vs_scenarioA_pct": 2})
M.to_csv("results/metrics.csv", index=False)
print("\n== METRIK (kWh/hari) ==")
print(M.drop(columns="params").to_string(index=False))

# ------------------- konfigurasi terbaik MENURUT VALIDATION (bukan test) untuk plot
vcand = M[(M.protocol == "validation") & (M.scenario.isin(SCEN))]
b = vcand.loc[vcand["mae"].idxmin()]
b_scen, b_model = b["scenario"], b["model"]
print(f"\nKonfigurasi terbaik di validation: skenario {b_scen}, model {b_model}")

# ------------------------------------------------------- analisis residual
pred = test_pred[(b_scen, b_model)]
res = y.loc[te] - pred
dow = pd.DataFrame({
    "day_of_week": res.index.dayofweek, "residual": res.values,
    "abs_residual": res.abs().values,
}).groupby("day_of_week").agg(n=("residual", "size"), bias=("residual", "mean"),
                              mae=("abs_residual", "mean")).round(3)
dow.index = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]
dow.to_csv("results/residual_by_dow.csv")
print("\n== Residual per hari dalam minggu (test, model terbaik validation) ==")
print(dow)

# ------------------------------------------------------------------ gambar
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(y.index, y.values, lw=0.7, color="#2a6f97")
for s in ("validation", "test"):
    ax.axvline(pd.Timestamp(SPLITS[s][0]), color="gray", ls="--", lw=0.8)
ax.set_ylabel("Energi harian (kWh)")
ax.set_title("Konsumsi energi harian (garis putus: awal validation dan test)")
fig.tight_layout()
fig.savefig("figures/fig_daily_energy.png", dpi=200)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
for ax, proto, title in zip(axes, ["validation", "test_walkforward"],
                            ["Validation", "Test (walk-forward)"]):
    g = M[M.protocol == proto]
    labels = [f"{r.scenario}\n{r.model}" if r.model != "-" else r.scenario
              for r in g.itertuples()]
    ax.bar(range(len(g)), g["mae"], color="#2a6f97")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("MAE (kWh/hari)")
    ax.set_title(title)
fig.tight_layout()
fig.savefig("figures/fig_mae_comparison.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 3.8))
ax.plot(te, y.loc[te], lw=1, label="Aktual", color="black")
ax.plot(te, X.loc[te, "lag7"], lw=0.8, label="Seasonal naive", color="#e07a5f", alpha=0.8)
ax.plot(te, pred, lw=0.9, label=f"Skenario {b_scen} ({b_model})", color="#2a6f97")
ax.set_ylabel("Energi harian (kWh)")
ax.set_title("Aktual vs prediksi pada periode test (walk-forward)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/fig_actual_vs_pred_test.png", dpi=200)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot(res.index, res.values, lw=0.7, color="#2a6f97")
axes[0].axhline(0, color="gray", lw=0.8)
axes[0].set_title("Residual terhadap waktu (test)")
axes[0].set_ylabel("Aktual - prediksi (kWh)")
axes[0].tick_params(axis="x", rotation=45)
axes[1].hist(res.values, bins=30, color="#2a6f97")
axes[1].set_title("Distribusi residual")
fig.tight_layout()
fig.savefig("figures/fig_residuals_test.png", dpi=200)
plt.close(fig)

# --------------------------------------------------------------- log reproduksi
log = {
    "dijalankan": datetime.now().isoformat(timespec="seconds"),
    "python": platform.python_version(), "pandas": pd.__version__,
    "numpy": np.__version__, "scikit-learn": sklearn.__version__,
    "seed": SEED, "n_random_seeds": N_RANDOM_SEEDS, "splits": SPLITS,
    "best_params": {f"{s}|{m}": p for (s, m), p in best.items()},
    "best_validation_config": [b_scen, b_model],
}
Path("results/run_log.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
print("\nSelesai. Keluaran: results/*.csv, results/run_log.json, figures/*.png")
