# Catatan Keputusan Metodologi (P4 — Peramalan Konsumsi Energi Rumah Tangga)

- **Tanggal keputusan dibuat:** 2026-11-24
- **Status:** TERKUNCI pada 2026-11-24, setelah hasil `src/check_nan_days.py` dicek (lihat bagian 5).
- **Aturan:** keputusan di dokumen ini dibuat **sebelum** melihat hasil evaluasi di data test. Setiap perubahan setelah ini dicatat di bagian Log perubahan beserta alasannya.

## 1. Sumber data

| Item | Isian |
|---|---|
| Dataset | Individual Household Electric Power Consumption |
| Sumber di Kaggle | https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set |
| Asal data primer | UCI Machine Learning Repository (Hébrail & Bérard) |
| Tanggal unduh | 12 November 2026 |
| Lisensi | Kaggle: "Database: Open Database, Contents: Database Contents" (Open Database License / ODbL untuk basis data, Database Contents License / DbCL v1.0 untuk isinya) |
| Berkas | `household_power_consumption.txt` (pemisah `;`, missing ditulis `?`) |

Hasil audit lengkap: `results/audit_summary.txt`.

## 2. Ringkasan hasil audit

- 2.075.259 baris per menit, 16 Des 2006 17:24 sampai 26 Nov 2010 21:02 (1.442 hari kalender).
- Tidak ada timestamp duplikat dan tidak ada menit yang absen dari file.
- Missing 1,252% (25.979 baris). Semua kolom hilang bersamaan, jadi yang bolong adalah baris penuh.
- 71 run missing; terpanjang 7.226 menit (sekitar 5 hari); 6 run berlangsung 1 hari atau lebih.
- 84 hari punya paling sedikit satu menit missing; 24 hari missing lebih dari 10% (termasuk 9 hari tanpa data sama sekali).

## 3. Definisi target

- Unit observasi: **hari**.
- `Global_active_power` berunit kW (rata-rata per menit), bukan kWh.
- Energi harian (kWh) = jumlah dari `Global_active_power / 60` atas semua menit valid dalam hari tersebut.

## 4. Kebijakan missing value

1. Hari dengan missing **10% atau kurang**: energi diskalakan ke 1.440 menit (`sum × 1440 / menit_valid`).
2. Hari dengan missing **lebih dari 10%**: dibiarkan NaN, **tidak diimputasi**.
3. Fitur lag dan rolling dibiarkan NaN bila jendelanya memuat hari NaN. Baris dengan fitur NaN dibuang dari evaluasi.
4. Semua baseline dan skenario dievaluasi pada **himpunan hari yang sama**, yaitu hari yang valid untuk semua skenario, agar perbandingan MAE/RMSE adil.
5. Jumlah hari yang terbuang di tiap split dicatat dan dilaporkan di bagian Methods paper.

Alasan: tidak ada nilai buatan yang ikut dinilai, dan tidak ada informasi dari masa depan yang bocor lewat interpolasi.

## 5. Pembagian data kronologis

| Bagian | Periode | Fungsi |
|---|---|---|
| Train | 16 Des 2006 – 31 Des 2008 | fitting model |
| Validation | 1 Jan – 31 Des 2009 | memilih parameter |
| Test | 1 Jan – 26 Nov 2010 | dilaporkan sekali di akhir |

Aturan:
- Tidak ada pengacakan untuk evaluasi utama. Random split hanya dipakai sebagai pembanding untuk menguji H2 dan diberi label jelas.
- Tuning parameter hanya memakai train dan validation. Test dihitung sekali setelah rancangan dibekukan.
- Pra-pemrosesan yang dipelajari dari data (misalnya scaler) hanya di-fit pada train.

Hasil cek sebaran NaN (`results/nan_check.txt`):

| Split | Jumlah hari | Hari NaN | Persen NaN |
|---|---|---|---|
| Train | 747 | 4 | 0,54 |
| Validation | 365 | 4 | 1,10 |
| Test | 330 | 16 | 4,85 |

Keputusan setelah cek: batas split dipertahankan. Hari NaN sedikit di semua split (maksimum 4,85% di test) dan penggeseran batas tidak mengurangi kebutuhan menangani missing. Hari pertama (16 Des 2006) dan hari terakhir (26 Nov 2010) berstatus NaN karena rekaman hari itu tidak penuh.

## 6. Pencegahan leakage pada fitur

- Target hari *t* hanya boleh diprediksi dari informasi hari ≤ *t*−1.
- Lag: `lag1 = d.shift(1)`, `lag7 = d.shift(7)`.
- Rolling mean selalu dihitung dari series yang sudah digeser: `d.shift(1).rolling(7).mean()` dan `d.shift(1).rolling(14).mean()`.

## 7. Desain eksperimen (ringkas)

| Komponen | Rancangan |
|---|---|
| Baseline | Naïve (kemarin) dan seasonal naïve (7 hari lalu) |
| Skenario A | Fitur kalender (hari dalam minggu, bulan) |
| Skenario B | A + lag 1 hari dan lag 7 hari |
| Skenario C | B + rolling mean 7 dan 14 hari |
| Metrik | MAE, RMSE, plot aktual vs prediksi, analisis residual |

Model yang sama dipakai di semua skenario; hanya fitur yang berubah.

## 8. Keterbatasan yang sudah diketahui

- Data berasal dari **satu rumah tangga**, sehingga generalisasi terbatas.
- Data test berhenti 26 Nov 2010 (hari terakhir hanya terekam sampai pukul 21:02 sehingga berstatus NaN), jadi test efektif berakhir 25 Nov 2010 dan musim dingin (Desember) hanya terwakili sebagian.
- Hari NaN lebih banyak di periode test (4,85%) dibanding train (0,54%) dan validation (1,10%). Karena fitur lag 7 hari dan rolling 14 hari ikut kosong setelah hari NaN, jumlah hari test yang benar-benar dievaluasi akan lebih sedikit; angka pastinya dilaporkan setelah eksperimen.
- Hari yang missing lebih dari 10% dikeluarkan dari evaluasi, sehingga hasil hanya berlaku untuk hari dengan data cukup lengkap.

## 9. Belum diputuskan

- Pilihan model (misalnya Ridge / Random Forest) dan seed.
- Jadwal walk-forward (frekuensi pelatihan ulang di periode test).

## 10. Log perubahan

| Tanggal | Perubahan | Alasan |
|---|---|---|
| 2026-11-24 | Dokumen dibuat | Keputusan awal setelah audit data |
| 2026-11-24 | Status dikunci; tabel sebaran NaN dan keputusan split diisi | Hasil `src/check_nan_days.py` menunjukkan tidak ada split yang bolong parah |
