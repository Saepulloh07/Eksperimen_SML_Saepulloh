# Eksperimen_SML_nama-siswa

Repository submission kelas **Membangun Sistem Machine Learning (MSML)** —
tahap *Eksperimen & Preprocessing Data* (Kriteria 1).

Proyek ini melakukan rekonsiliasi data keuangan klinik dari 4 sumber
(kasir tunai, transfer, closing sistem, dan mutasi rekening/arus kas),
mendeteksi ketidaksesuaian (selisih/anomali) antar sumber, lalu
menghasilkan dataset berlabel yang siap dipakai untuk melatih model
klasifikasi.

## Struktur Folder

```
Eksperimen_SML_nama-siswa/
├── .github/
│   └── workflows/
│       └── preprocessing.yml        # GitHub Actions: preprocessing otomatis
├── namadataset_raw/                 # Data mentah (4 sumber Excel)
│   └── README.md
├── preprocessing/
│   ├── Eksperimen_nama-siswa.ipynb  # Notebook eksperimen (eksplorasi + preprocessing manual)
│   ├── automate_nama-siswa.py       # Script preprocessing otomatis (fungsi modular)
│   └── namadataset_preprocessing/   # Output hasil preprocessing (auto-generated)
├── requirements.txt
└── README.md
```

## Alur Preprocessing

1. **Load data mentah** — baca 4 sumber Excel (kasir/TO, transfer/TF, closing/CL, arus kas/AK).
2. **Pembersihan per sumber** — perbaikan header, parsing tanggal, konversi nominal ke numerik, standardisasi nama kolom.
3. **Fuzzy name matching & rekonsiliasi** — menyamakan penulisan nama pasien antar sumber, lalu menggabungkan (outer join) semua sumber berdasarkan tanggal & nama pasien.
4. **Pelabelan multi-kelas** — setiap baris diberi label kategori hasil audit (mis. `Cleared & Matched`, `Fraud Risk`, `Unrecorded`, `Split Payment`, dll) berdasarkan aturan bisnis.
5. **Finalisasi untuk ML**:
   - Penanganan missing value & duplikat
   - Feature engineering dari tanggal (hari, hari dalam minggu, akhir pekan)
   - Deteksi outlier (IQR) pada nominal transaksi
   - Encoding kategorikal (`LabelEncoder`)
   - Scaling numerik (`StandardScaler`)
   - Train-test split (80/20, stratified bila memungkinkan)
6. **Simpan artefak** — `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, dataset lengkap, `scaler.joblib`, `label_encoder_*.joblib`, dan `label_mapping.json`.

## Menjalankan Preprocessing Secara Lokal

```bash
pip install -r requirements.txt

python preprocessing/automate_nama-siswa.py \
  --raw_dir namadataset_raw \
  --output_dir preprocessing/namadataset_preprocessing
```

Argumen opsional (jika nama file/sheet berbeda periode):

```bash
python preprocessing/automate_nama-siswa.py \
  --raw_dir namadataset_raw \
  --output_dir preprocessing/namadataset_preprocessing \
  --laporan_kasir_file "Laporan kasir.xlsx" \
  --closing_file "data sistem 1 - 30 juni.xlsx" \
  --closing_sheet "PendapatanPerAkunClosing" \
  --aruskas_file "ARUS KAS 2026.xlsx" \
  --aruskas_sheet "Juni"
```

## Otomatisasi via GitHub Actions

Workflow `.github/workflows/preprocessing.yml` akan otomatis berjalan saat:
- Ada perubahan (push) pada folder `namadataset_raw/` atau file `automate_nama-siswa.py`
- Dipicu manual lewat tab **Actions > Automated Data Preprocessing > Run workflow**
- Terjadwal tiap awal bulan (dapat diubah/dihapus sesuai kebutuhan)

Setiap kali berjalan, workflow akan meng-*commit* ulang dataset hasil
preprocessing terbaru ke `preprocessing/namadataset_preprocessing/`, dan
juga mengunggahnya sebagai *artifact* yang bisa diunduh dari halaman run
Actions tersebut.
