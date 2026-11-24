# Peramalan Konsumsi Listrik Harian Rumah Tangga: Fitur Lag dan Rata-rata Bergulir

Proyek TTKI (PENS, Semester 5), topik P4 *Energy informatics: household electricity forecasting*.
Repositori ini berisi kode, hasil, dan catatan keputusan untuk paper IMRAD
**"Rekayasa Fitur Berbasis Lag untuk Peramalan Jangka Pendek Konsumsi Listrik Rumah Tangga"**.

## Tujuan dan pertanyaan penelitian

Memprediksi energi harian (kWh) satu rumah tangga dari data per menit dan mengukur seberapa besar fitur historis membantu.

- **RQ1**: seberapa besar fitur lag dan rata-rata bergulir menurunkan galat dibanding fitur kalender saja dan dibanding baseline naïve dan seasonal naïve?
- **RQ2**: apakah random split memberi gambaran manfaat fitur historis yang berbeda dibanding validasi kronologis walk-forward?

Desain singkat:

| Komponen | Isi |
|---|---|
| Baseline | naïve (nilai kemarin), seasonal naïve (nilai 7 hari lalu) |
| Skenario A | fitur kalender (hari dalam minggu, bulan) |
| Skenario B | A + lag 1 dan lag 7 |
| Skenario C | B + rata-rata bergulir 7 dan 14 hari |
| Model | Ridge Regression, Random Forest (300 pohon) |
| Split | kronologis: train 16 Des 2006 - 31 Des 2008, validation 2009, test 1 Jan - 26 Nov 2010 |
| Evaluasi test | walk-forward, dilatih ulang tiap bulan, hyperparameter tetap dari validation |
| Metrik | MAE dan RMSE (kWh/hari) |

Semua keputusan metodologi, termasuk kebijakan data hilang, dicatat sebelum data uji dievaluasi di [`docs/decisions.md`](docs/decisions.md).

## Sumber data, lisensi, dan atribusi

- Dataset: *Individual Household Electric Power Consumption* (Hébrail & Bérard), UCI Machine Learning Repository, DOI [10.24432/C58K54](https://doi.org/10.24432/C58K54).
- Salinan yang dipakai: Kaggle `uciml/electric-power-consumption-data-set` (https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set), diunduh pada 12 November 2026.
- Lisensi yang tertera di Kaggle: *Database: Open Database License; Contents: Database Contents License*. Data dipakai dengan atribusi kepada pembuat asli dan **tidak disertakan** dalam repositori ini.
- Isi dataset: 2.075.259 pengukuran per menit dari satu rumah tangga, 16 Des 2006 - 26 Nov 2010; sekitar 1,25% baris hilang (ditulis `?`).

## Struktur repositori

```
.
├── data/
│   ├── raw/                  # TIDAK ada di Git; taruh household_power_consumption.txt di sini
│   └── daily_energy.csv      # keluaran audit_data.py (energi harian)
├── docs/decisions.md         # keputusan desain dan kebijakan data hilang
├── figures/                  # gambar hasil eksperimen (PNG)
├── references/literature_matrix.csv
├── results/                  # metrics.csv, eval_counts.csv, dataset_table.csv, dll.
├── src/
│   ├── audit_data.py         # audit data + energi harian
│   ├── check_nan_days.py     # sebaran hari NaN per tahun dan per split
│   └── run_experiments.py    # fitur, tuning, walk-forward, random split, gambar
├── requirements.txt
└── environment-freeze.txt
```

## Lingkungan

Diuji dengan Python 3.14.0, pandas 3.1.1, numpy 2.5.5, scikit-learn 1.9.1, matplotlib 3.11.2. Versi lengkap ada di `environment-freeze.txt`.

```bash
python -m venv .venv
source .venv/Scripts/activate      # Git Bash di Windows; Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Cara mereproduksi

Jalankan dari **root repositori**, berurutan.

1. Unduh dataset dari Kaggle (lihat tautan di atas), ekstrak, dan letakkan `household_power_consumption.txt` di `data/raw/`.
2. Audit data dan buat energi harian:
   ```bash
   python src/audit_data.py data/raw/household_power_consumption.txt
   ```
   Menghasilkan `data/daily_energy.csv` dan `results/audit_summary.txt`.
3. Periksa sebaran hari NaN per split (opsional, mendukung keputusan split):
   ```bash
   python src/check_nan_days.py
   ```
4. Jalankan eksperimen utama:
   ```bash
   python src/run_experiments.py
   ```
   Menghasilkan `results/*.csv`, `results/run_log.json`, dan empat gambar di `figures/`.

Hasil bersifat deterministik: seed acak 42 untuk model, dan seed 42-46 untuk perbandingan random split.

## Ringkasan hasil

Pada data uji (258 hari dievaluasi dari 330 hari), MAE terendah adalah 4,300 kWh/hari (Random Forest, Skenario C), dibanding 5,234 (naïve) dan 6,101 (seasonal naïve). Fitur lag dan rata-rata bergulir menurunkan MAE 3,1% sampai 8,7% terhadap kalender saja. Random split tidak menunjukkan manfaat yang konsisten lebih besar. Angka lengkap ada di `results/metrics.csv`; interpretasi dan keterbatasan ada di paper.

## Pengendalian kebocoran informasi

- Target hari *t* hanya diprediksi dari informasi hari *t*-1 ke belakang; rata-rata bergulir dihitung dari deret yang digeser satu hari.
- Hari dengan data hilang lebih dari 10% dibiarkan NaN, tidak diimputasi; fitur yang jendelanya memuat NaN dibiarkan kosong.
- Semua baseline dan skenario dinilai pada himpunan hari evaluasi yang sama.
- Hyperparameter dipilih hanya pada validation; data uji dihitung satu kali.

## Keterbatasan

Satu rumah tangga, satu periode uji, tanpa uji signifikansi, tanpa variabel eksternal (cuaca, hari libur). Detail ada di bagian Pembahasan paper.

## Penggunaan AI

Asisten AI (Claude, Anthropic) dipakai untuk membantu merancang struktur proyek, menulis dan memeriksa kode, mencocokkan kutipan literatur, dan menyusun draf naskah. Seluruh kode dijalankan oleh penulis dan semua angka berasal dari eksperimen penulis. Pernyataan lengkap ada di paper.

## Sitasi dataset

Hébrail, G., & Bérard, A. (2012). *Individual household electric power consumption* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54

## Penulis

[Nama penulis], Program Sarjana Terapan Teknik Informatika, Politeknik Elektronika Negeri Surabaya (PENS).
