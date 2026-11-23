"""Audit dataset Individual Household Electric Power Consumption (Kaggle/UCI).

Pemakaian:
    python audit_data.py path/ke/household_power_consumption.txt

Output:
    results/audit_summary.txt   -> ringkasan audit (masukkan ke paper/README)
    data/daily_energy.csv       -> energi harian (kWh) + jumlah menit valid per hari
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

path = Path(sys.argv[1])
Path("results").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)
out = []


def log(msg=""):
    print(msg)
    out.append(str(msg))


# 1. Baca: pemisah ';', missing ditulis '?'
df = pd.read_csv(path, sep=";", na_values=["?", ""], low_memory=False)
df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
)
df = df.drop(columns=["Date", "Time"]).set_index("datetime").sort_index()
num_cols = [c for c in df.columns]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

log("== 1. Ukuran & rentang waktu ==")
log(f"Baris: {len(df):,} | Kolom: {list(df.columns)}")
log(f"Rentang: {df.index.min()} s.d. {df.index.max()}")

# 2. Missing values
log("\n== 2. Missing values ==")
miss = df.isna().sum()
log(pd.DataFrame({"missing": miss, "persen": (miss / len(df) * 100).round(3)}))

# 3. Timestamp: duplikat & menit yang hilang sama sekali dari file
log("\n== 3. Timestamp ==")
log(f"Timestamp duplikat: {df.index.duplicated().sum():,}")
full = pd.date_range(df.index.min(), df.index.max(), freq="min")
log(f"Menit seharusnya: {len(full):,} | ada di file: {len(df):,} | "
    f"baris tidak ada: {len(full) - len(df):,}")
df = df[~df.index.duplicated(keep="first")].reindex(full)  # menit absen -> NaN

# 4. Run missing terpanjang pada Global_active_power
gap = df["Global_active_power"].isna()
runs = gap.ne(gap.shift()).cumsum()[gap]
if len(runs):
    run_len = runs.groupby(runs).size()
    log("\n== 4. Run missing Global_active_power ==")
    log(f"Jumlah run: {len(run_len):,} | terpanjang: {run_len.max():,} menit "
        f"({run_len.max() / 60:.1f} jam)")
    log(f"Run >= 1 jam: {(run_len >= 60).sum():,} | >= 1 hari: {(run_len >= 1440).sum():,}")

# 5. Statistik & nilai ganjil (daya tidak boleh negatif)
log("\n== 5. Statistik ringkas ==")
log(df.describe().T[["min", "mean", "max"]].round(3))
log(f"Global_active_power < 0: {(df['Global_active_power'] < 0).sum():,}")

# 6. Energi harian. kW rata-rata per menit -> kWh = kW / 60
df["kwh_menit"] = df["Global_active_power"] / 60.0
daily = df["kwh_menit"].resample("D").agg(["sum", "count"])
daily.columns = ["energy_kwh_raw", "valid_minutes"]
daily["missing_minutes"] = 1440 - daily["valid_minutes"]
daily["complete"] = daily["missing_minutes"] == 0
log("\n== 6. Kelengkapan harian ==")
log(f"Jumlah hari: {len(daily):,} | lengkap 1440 menit: {daily['complete'].sum():,}")
log(f"Hari dengan >0 menit missing: {(~daily['complete']).sum():,}")
log(f"Hari dengan >10% menit missing: {(daily['missing_minutes'] > 144).sum():,}")
log(f"Hari tanpa data sama sekali: {(daily['valid_minutes'] == 0).sum():,}")

# Kebijakan sementara (TETAPKAN & dokumentasikan sendiri sebelum melihat hasil test!):
# hari dengan <=10% menit missing -> skala sum ke 1440 menit; lainnya -> NaN.
ok = daily["missing_minutes"] <= 144
daily["energy_kwh"] = np.where(
    ok & (daily["valid_minutes"] > 0),
    daily["energy_kwh_raw"] * 1440 / daily["valid_minutes"].replace(0, np.nan),
    np.nan,
)
log(f"Hari NaN setelah kebijakan: {daily['energy_kwh'].isna().sum():,}")
daily.to_csv("data/daily_energy.csv", index_label="date")
Path("results/audit_summary.txt").write_text("\n".join(out), encoding="utf-8")
log("\nTersimpan: data/daily_energy.csv, results/audit_summary.txt")
