"""Langkah 2c: cek sebaran hari NaN per tahun dan per split.

Jalankan dari root repo (setelah audit_data.py menghasilkan data/daily_energy.csv):
    python src/check_nan_days.py

Output: tercetak di layar + results/nan_check.txt
"""
from pathlib import Path

import pandas as pd

SPLITS = {
    "train": ("2006-12-16", "2008-12-31"),
    "validation": ("2009-01-01", "2009-12-31"),
    "test": ("2010-01-01", "2010-11-26"),
}

Path("results").mkdir(exist_ok=True)
out = []


def log(msg=""):
    print(msg)
    out.append(str(msg))


d = pd.read_csv("data/daily_energy.csv", index_col="date", parse_dates=True)
nan_days = d[d["energy_kwh"].isna()]

log("== NaN per tahun ==")
log(nan_days.groupby(nan_days.index.year).size().to_string())

log("\n== Ringkasan per split ==")
rows = []
for name, (start, end) in SPLITS.items():
    part = d.loc[start:end]
    n_nan = part["energy_kwh"].isna().sum()
    rows.append({
        "split": name,
        "hari": len(part),
        "hari_NaN": int(n_nan),
        "persen_NaN": round(n_nan / len(part) * 100, 2),
    })
log(pd.DataFrame(rows).to_string(index=False))

log("\n== Daftar hari NaN (valid_minutes = menit yang berhasil terekam) ==")
log(nan_days[["valid_minutes"]].to_string())

Path("results/nan_check.txt").write_text("\n".join(out), encoding="utf-8")
log("\nTersimpan: results/nan_check.txt")
