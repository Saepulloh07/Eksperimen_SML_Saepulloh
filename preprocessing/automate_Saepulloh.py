"""
automate_nama-siswa.py
=======================
Script otomatisasi preprocessing data keuangan klinik (rekonsiliasi kasir,
transfer, closing, dan arus kas) menjadi dataset siap latih untuk model
Machine Learning (klasifikasi kategori hasil audit/rekonsiliasi).

Pipeline ini adalah hasil konversi dari notebook eksperimen
`Eksperimen_nama-siswa.ipynb` — tahapannya SAMA PERSIS dengan notebook,
namun disusun ulang menjadi fungsi-fungsi modular agar bisa dijalankan
otomatis (mis. lewat CLI atau GitHub Actions) tanpa perlu Google Colab
atau Google Drive.

Cara pakai (CLI):
    python automate_Saepulloh.py \
        --raw_dir keuangan_raw \
        --output_dir preprocessing/keuangan_preprocessing

Cara pakai (sebagai modul):
    from automate_Saepulloh import preprocess_data
    hasil = preprocess_data(raw_dir="keuangan_raw", output_dir="preprocessing/keuangan_preprocessing")
    X_train, X_test, y_train, y_test = hasil["X_train"], hasil["X_test"], hasil["y_train"], hasil["y_test"]
"""

import os
import re
import json
import argparse
import difflib

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# =====================================================================
# 1. LOAD RAW DATA
# =====================================================================
def load_raw_data(
    raw_dir,
    laporan_kasir_file="Laporan kasir.xlsx",
    closing_file="data sistem 1 - 31 mei.xlsx",
    closing_sheet="PendapatanPerAkunClosing",
    aruskas_file="ARUS KAS 2026.xlsx",
    aruskas_sheet="Mei",
):
    """
    Membaca 4 sumber data mentah dari folder `raw_dir`:
      - Sheet 'TO'          pada `laporan_kasir_file`  -> df_to
      - Sheet 'TRANSFERAN'  pada `laporan_kasir_file`  -> df_tf
      - Sheet `closing_sheet` pada `closing_file`      -> df_cl
      - Sheet `aruskas_sheet` pada `aruskas_file`       -> df_ak

    Semua nama file/sheet dapat dikustomisasi lewat argumen fungsi karena
    berbeda periode (bulan) biasanya berarti nama file Excel yang berbeda.
    """
    laporan_kasir_path = os.path.join(raw_dir, laporan_kasir_file)
    closing_path = os.path.join(raw_dir, closing_file)
    aruskas_path = os.path.join(raw_dir, aruskas_file)

    for path in [laporan_kasir_path, closing_path, aruskas_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File data mentah tidak ditemukan: {path}")

    df_to = pd.read_excel(laporan_kasir_path, sheet_name="TO")
    df_tf = pd.read_excel(laporan_kasir_path, sheet_name="TRANSFERAN")
    df_cl = pd.read_excel(closing_path, sheet_name=closing_sheet)
    df_ak = pd.read_excel(aruskas_path, sheet_name=aruskas_sheet)

    print(f"[load_raw_data] df_to: {df_to.shape}, df_tf: {df_tf.shape}, "
          f"df_cl: {df_cl.shape}, df_ak: {df_ak.shape}")

    return df_to, df_tf, df_cl, df_ak


# =====================================================================
# 2. PEMBERSIHAN PER SUMBER DATA (identik dengan notebook eksperimen)
# =====================================================================
def clean_df_to(df_to):
    """Membersihkan data 'TO' (kasir tunai)."""
    df_to = df_to.iloc[1:].reset_index(drop=True)
    df_to.columns = df_to.iloc[1]
    df_to = df_to.iloc[3:].reset_index(drop=True)

    new_column_names = [
        "Tanggal", "No", "Nama Pasien", "Konsul", "Adm",
        "USG (2/4) D", "T.Verban", "Injeksi", "USG V", "Total",
    ]
    if len(new_column_names) != len(df_to.columns):
        raise ValueError(
            f"Jumlah kolom baru ({len(new_column_names)}) tidak sesuai dengan "
            f"jumlah kolom df_to ({len(df_to.columns)}). Cek struktur sheet 'TO'."
        )
    df_to.columns = new_column_names

    df_to = df_to[["Tanggal", "Nama Pasien", "Total"]].copy()
    df_to["Tanggal"] = pd.to_datetime(df_to["Tanggal"], errors="coerce")
    df_to["Tanggal"] = df_to["Tanggal"].dt.strftime("%d-%m-%Y")
    df_to["Tanggal"] = df_to["Tanggal"].ffill()

    df_to = df_to.dropna(subset=["Nama Pasien", "Total"])
    df_to = df_to.rename(columns={"Total": "Nominal"})

    print(f"[clean_df_to] hasil akhir: {df_to.shape}")
    return df_to


def clean_df_tf(df_tf):
    """Membersihkan data 'TRANSFERAN'."""
    new_column_names = [
        "No", "Tanggal Transaksi", "Nama Pasien", "Poli OBGYN", "Poli DVE",
        "Poli Anak", "IGD", "Ranap IBU & Bayi", "Ranap Anak", "Lainnya", "KET",
    ]
    if len(new_column_names) != len(df_tf.columns):
        raise ValueError(
            f"Jumlah kolom baru ({len(new_column_names)}) tidak sesuai dengan "
            f"jumlah kolom df_tf ({len(df_tf.columns)}). Cek struktur sheet 'TRANSFERAN'."
        )
    df_tf.columns = new_column_names
    df_tf = df_tf.iloc[3:].reset_index(drop=True)

    df_tf = df_tf.drop(columns=["No", "KET"])

    nominal_columns = [
        "Poli OBGYN", "Poli DVE", "Poli Anak", "IGD",
        "Ranap IBU & Bayi", "Ranap Anak", "Lainnya",
    ]
    for col in nominal_columns:
        df_tf[col] = pd.to_numeric(df_tf[col], errors="coerce").fillna(0)

    df_tf["Nominal"] = df_tf[nominal_columns].sum(axis=1)
    df_tf = df_tf.drop(columns=nominal_columns)

    df_tf["Tanggal Transaksi"] = pd.to_datetime(df_tf["Tanggal Transaksi"], errors="coerce")
    df_tf["Tanggal Transaksi"] = df_tf["Tanggal Transaksi"].dt.strftime("%d-%m-%Y")

    df_tf = df_tf.iloc[:-2].copy()
    df_tf["Tanggal Transaksi"] = df_tf["Tanggal Transaksi"].ffill()
    df_tf["Nominal"] = df_tf["Nominal"].astype(int)

    df_tf = df_tf.rename(columns={"Tanggal Transaksi": "Tanggal"})

    print(f"[clean_df_tf] hasil akhir: {df_tf.shape}")
    return df_tf


def clean_df_cl(df_cl):
    """Membersihkan data 'Closing'."""
    df_cl = df_cl.rename(columns={"Unnamed: 6": "NOMINAL"})
    df_cl = df_cl.drop(columns=["No.", "No.Rawat/No.Nota"])

    df_cl["Tanggal"] = pd.to_datetime(df_cl["Tanggal"], errors="coerce")
    df_cl["Tanggal"] = df_cl["Tanggal"].dt.strftime("%d-%m-%Y")

    df_cl = df_cl.iloc[:-10]
    df_cl = df_cl.ffill()

    def _clean_keterangan(text):
        if pd.isna(text):
            return text
        text_str = str(text)
        if re.match(r"^(by\.)\s+", text_str, re.IGNORECASE):
            return text_str
        return re.sub(r"^(ny\.|an\.|tn\.|nn\.)\s*", "", text_str, flags=re.IGNORECASE)

    df_cl["Nama Pasien"] = df_cl["Nama Pasien"].apply(_clean_keterangan)

    df_cl = df_cl[df_cl["Akun Closing"] != "2. Piutang BPJS"]
    df_cl = df_cl[df_cl["Akun Closing"] != "PIUTANG OBAT & BHP"]
    df_cl = df_cl[df_cl["Nama Pasien"] != "x"]
    df_cl = df_cl[~df_cl["Jenis/Cara Bayar"].str.contains("Penjualan Apotek", na=False)]

    print(f"[clean_df_cl] hasil akhir: {df_cl.shape}")
    return df_cl


def clean_df_ak(df_ak):
    """Membersihkan data 'Arus Kas'."""
    columns_to_drop = [
        "No.", "Tanggal Buku", "Kode Akun", "CASH", "Unnamed: 7", "Saldo",
        "GIRO", "Unnamed: 10", "Saldo.1", "Tanggal Transaksi.1", "BISNIS UMUM",
        "Unnamed: 14", "SALDO",
    ]
    df_ak = df_ak.drop(columns=[c for c in columns_to_drop if c in df_ak.columns])
    df_ak = df_ak.rename(columns={"Unnamed: 5": "Nominal"})

    df_ak = df_ak.dropna(how="all").copy()

    df_ak["Tanggal Transaksi"] = pd.to_datetime(df_ak["Tanggal Transaksi"], errors="coerce")
    df_ak["Tanggal Transaksi"] = df_ak["Tanggal Transaksi"].dt.strftime("%d-%m-%Y").fillna("")
    df_ak["Nominal"] = pd.to_numeric(df_ak["Nominal"], errors="coerce").astype("Int64")

    identification_pattern = r"a/n|Rajal Kebidanan|Rajal IGD|Poli Anak|Rajal DVE|Vaksin"
    df_pasien = df_ak[df_ak["KETERANGAN"].astype(str).str.contains(
        identification_pattern, case=False, na=False
    )].copy()

    removal_pattern = r"(?:a/n\.?|Rajal Kebidanan|Rajal IGD|Poli Anak|Rajal DVE|Vaksin)\s*"
    df_pasien["KETERANGAN"] = df_pasien["KETERANGAN"].astype(str).apply(
        lambda x: re.sub(removal_pattern, "", x, flags=re.IGNORECASE).strip()
    )
    df_pasien = df_pasien[df_pasien["KETERANGAN"] != ""]

    hasil = df_pasien[["Tanggal Transaksi", "KETERANGAN", "Nominal"]].copy()
    hasil.columns = ["Tanggal Transaksi", "Keterangan", "Nominal"]

    def _clean_keterangan_ak(text):
        if pd.isna(text):
            return text
        text = str(text).strip()
        removal_regex = (
            r"^(?:"
            r"Ranap Naik Kelas\s*|"
            r"laboratorium\s*|"
            r"Imunisasi\s*|"
            r"Ranap Umum\s*|"
            r"Poli Bedah\s*|"
            r"Poli Kulit\s*|"
            r"a\s*/\s*n\.?\s*|"
            r"ny\.?(?=\s|[A-Z]|$)\s*|"
            r"tn\.?(?=\s|[A-Z]|$)\s*|"
            r"an\.?(?=\s)\s*"
            r")+"
        )
        return re.sub(removal_regex, "", text, flags=re.IGNORECASE).strip()

    hasil["Keterangan"] = hasil["Keterangan"].apply(_clean_keterangan_ak)
    hasil = hasil.rename(columns={"Tanggal Transaksi": "Tanggal", "Keterangan": "Nama Pasien"})

    print(f"[clean_df_ak] hasil akhir: {hasil.shape}")
    return hasil


# =====================================================================
# 3. FUZZY NAME MATCHING
# =====================================================================
def cek_kemiripan_nama(n1, n2):
    """
    Mendeteksi apakah 2 nama merujuk ke orang yang sama.
    Contoh True: "M. ARVAN" vs "MUHAMMAD ARVAN"
    """
    if pd.isna(n1) or pd.isna(n2):
        return False

    n1_clean = re.sub(r"[^A-Z\s]", "", str(n1).upper()).strip()
    n2_clean = re.sub(r"[^A-Z\s]", "", str(n2).upper()).strip()

    if n1_clean == n2_clean:
        return True

    if difflib.SequenceMatcher(None, n1_clean, n2_clean).ratio() > 0.8:
        return True

    if n1_clean and n2_clean and n1_clean[0] == n2_clean[0]:
        tok1 = set(t for t in n1_clean.split() if len(t) > 2)
        tok2 = set(t for t in n2_clean.split() if len(t) > 2)
        if tok1.intersection(tok2):
            return True

    return False


# =====================================================================
# 4. BANGUN audit_df: REKONSILIASI + PELABELAN MULTI-KELAS
# =====================================================================
def build_audit_dataframe(df_to, df_tf, df_cl, df_ak):
    """
    Menggabungkan 4 sumber data (kasir/TO, transfer/TF, closing/CL, arus
    kas/AK) menjadi satu dataframe rekonsiliasi (`audit_df`) lengkap dengan
    label kategori hasil audit (`Label_Kategori` & `Target_ML`).

    Mengembalikan (audit_df, unik_labels) di mana `unik_labels` adalah
    pemetaan indeks -> nama label dari `pd.factorize`.
    """
    df_cl = df_cl.rename(columns={"NOMINAL": "Nominal CL", "Jenis/Cara Bayar": "Jenis Bayar"})
    df_ak = df_ak.rename(columns={"Nominal": "Nominal AK"})
    df_tf = df_tf.rename(columns={"Nominal": "Nominal TF"})
    df_to = df_to.rename(columns={"Nominal": "Nominal TO"})

    if "Akun Closing" not in df_cl.columns:
        df_cl["Akun Closing"] = "UNKNOWN"

    for df in [df_cl, df_ak, df_tf, df_to]:
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True, errors="coerce")
        df["Nama Pasien"] = df["Nama Pasien"].astype(str).str.upper().str.strip()

    df_cl["Jenis Bayar"] = df_cl["Jenis Bayar"].astype(str).str.upper().str.strip()
    df_cl["Akun Closing"] = df_cl["Akun Closing"].astype(str).str.upper().str.strip()

    df_cl["Nominal CL"] = pd.to_numeric(df_cl["Nominal CL"], errors="coerce")
    df_ak["Nominal AK"] = pd.to_numeric(df_ak["Nominal AK"], errors="coerce")
    df_tf["Nominal TF"] = pd.to_numeric(df_tf["Nominal TF"], errors="coerce")
    df_to["Nominal TO"] = pd.to_numeric(df_to["Nominal TO"], errors="coerce")

    # ---- Fuzzy matching nama pasien antar sumber (disamakan ke penulisan Closing) ----
    dict_cl_names = df_cl.groupby("Tanggal")["Nama Pasien"].apply(list).to_dict()

    for df_mutasi in [df_ak, df_tf, df_to]:
        for idx, row in df_mutasi.iterrows():
            tgl = row["Tanggal"]
            nama_mutasi = row["Nama Pasien"]
            if tgl in dict_cl_names:
                for nama_closing in dict_cl_names[tgl]:
                    if cek_kemiripan_nama(nama_mutasi, nama_closing):
                        df_mutasi.at[idx, "Nama Pasien"] = nama_closing
                        break

    # ---- Cross-date lookup untuk kasus pasien telat/tidak sinkron tanggal ----
    dict_cl_dates = df_cl.groupby("Nama Pasien")["Tanggal"].apply(set).to_dict()
    df_all_mutasi = pd.concat([
        df_ak[["Tanggal", "Nama Pasien"]],
        df_tf[["Tanggal", "Nama Pasien"]],
        df_to[["Tanggal", "Nama Pasien"]],
    ]).drop_duplicates()
    dict_mutasi_dates = df_all_mutasi.groupby("Nama Pasien")["Tanggal"].apply(set).to_dict()

    # ---- Merging seluruh sumber (outer join) ----
    audit_df = pd.merge(df_cl, df_ak[["Tanggal", "Nama Pasien", "Nominal AK"]],
                         on=["Tanggal", "Nama Pasien"], how="outer")
    audit_df = pd.merge(audit_df, df_tf[["Tanggal", "Nama Pasien", "Nominal TF"]],
                         on=["Tanggal", "Nama Pasien"], how="outer")
    audit_df = pd.merge(audit_df, df_to[["Tanggal", "Nama Pasien", "Nominal TO"]],
                         on=["Tanggal", "Nama Pasien"], how="outer")

    for col in ["Nominal CL", "Nominal AK", "Nominal TF", "Nominal TO"]:
        audit_df[col] = audit_df[col].fillna(0)

    audit_df["Exist_CL"] = np.where(audit_df["Nominal CL"] > 0, 1, 0)
    audit_df["Exist_AK"] = np.where(audit_df["Nominal AK"] > 0, 1, 0)
    audit_df["Exist_TF"] = np.where(audit_df["Nominal TF"] > 0, 1, 0)
    audit_df["Exist_TO"] = np.where(audit_df["Nominal TO"] > 0, 1, 0)
    audit_df["Jumlah_Titik_Mutasi"] = audit_df["Exist_AK"] + audit_df["Exist_TF"] + audit_df["Exist_TO"]

    audit_df["Jenis Bayar"] = audit_df["Jenis Bayar"].replace(["NAN", "NONE", "NULL"], "")
    audit_df["Jenis Bayar"] = audit_df["Jenis Bayar"].fillna("")
    audit_df.loc[(audit_df["Jenis Bayar"] == "") & (audit_df["Exist_TF"] == 1), "Jenis Bayar"] = "TRANSFER"
    audit_df.loc[(audit_df["Jenis Bayar"] == "") & (audit_df["Exist_TO"] == 1), "Jenis Bayar"] = "TO"
    audit_df.loc[(audit_df["Jenis Bayar"] == "") & (audit_df["Exist_AK"] == 1), "Jenis Bayar"] = "TUNAI"

    audit_df["Total_Mutasi"] = audit_df["Nominal AK"] + audit_df["Nominal TF"] + audit_df["Nominal TO"]
    audit_df["Selisih_Final"] = audit_df["Nominal CL"] - audit_df["Total_Mutasi"]

    def _assign_audit_label(row):
        nama = row["Nama Pasien"]
        tgl = row["Tanggal"]
        jb = str(row["Jenis Bayar"]).upper()
        selisih = row["Selisih_Final"]
        titik_mutasi = row["Jumlah_Titik_Mutasi"]

        if row["Exist_CL"] == 0:
            if nama in dict_cl_dates and any(d != tgl for d in dict_cl_dates[nama]):
                return "Tanggal Closing Tidak Sesuai (Uang masuk, tapi Closing di tanggal beda)"
            if row["Exist_TF"] == 1:
                return "Unrecorded: Uang Masuk TF, Tanpa Closing"
            if row["Exist_TO"] == 1:
                return "Unrecorded: Uang Masuk TO, Tanpa Closing"
            if row["Exist_AK"] == 1:
                return "Unrecorded: Uang Masuk Kasir, Tanpa Closing"
            return "Unrecorded: Multi-Mutasi"

        if titik_mutasi == 0:
            if nama in dict_mutasi_dates and any(d != tgl for d in dict_mutasi_dates[nama]):
                return "Pasien Telat Closing / Tanggal Mutasi Tidak Sesuai"
            return "Critical: Ada di Closing, Tidak ada di arus kas"

        if titik_mutasi == 1 and selisih == 0:
            if ("TRANS" in jb or "TF" in jb) and row["Exist_TF"] == 0:
                return "Fraud Risk: Klaim TF, tapi masuk kas lain"
            if "TO" in jb and row["Exist_TO"] == 0:
                return "Fraud Risk: Klaim TO, tapi masuk kas lain"
            if ("TUNAI" in jb or "CASH" in jb) and row["Exist_AK"] == 0:
                return "Fraud Risk: Klaim Tunai, tapi masuk kas lain"

        if titik_mutasi > 1:
            if selisih == 0:
                return "Match via Split Payment"
            elif selisih < 0:
                return "Anomaly: Double Input"
            else:
                return "Split Payment: Kurang Bayar"

        if selisih > 0:
            return "Kurang Bayar (Nominal CL > Mutasi)"
        if selisih < 0:
            return "Lebih Bayar (Nominal CL < Mutasi)"

        return "Cleared & Matched"

    audit_df["Label_Kategori"] = audit_df.apply(_assign_audit_label, axis=1)
    audit_df["Target_ML"], unik_labels = pd.factorize(audit_df["Label_Kategori"])

    kolom_final = [
        "Tanggal", "Nama Pasien", "Jenis Bayar", "Akun Closing",
        "Nominal CL", "Nominal AK", "Nominal TF", "Nominal TO",
        "Total_Mutasi", "Selisih_Final", "Jumlah_Titik_Mutasi",
        "Label_Kategori", "Target_ML",
    ]
    kolom_tersedia = [c for c in kolom_final if c in audit_df.columns]
    audit_df = audit_df[kolom_tersedia].sort_values(by=["Tanggal", "Nama Pasien"]).reset_index(drop=True)

    print(f"[build_audit_dataframe] audit_df: {audit_df.shape}")
    print(audit_df["Label_Kategori"].value_counts())

    return audit_df, unik_labels


# =====================================================================
# 5. FINALISASI PREPROCESSING UNTUK MODELING
# =====================================================================
def handle_missing_and_duplicates(audit_df):
    """Penanganan missing value & data duplikat (final)."""
    audit_df = audit_df.copy()

    audit_df["Nama Pasien"] = audit_df["Nama Pasien"].fillna("TIDAK DIKETAHUI")
    audit_df["Jenis Bayar"] = audit_df["Jenis Bayar"].astype(str).replace(
        ["", "nan", "NAN", "None", "NONE"], np.nan
    ).fillna("TIDAK DIKETAHUI")
    if "Akun Closing" in audit_df.columns:
        audit_df["Akun Closing"] = audit_df["Akun Closing"].fillna("TIDAK DIKETAHUI")

    kolom_numerik = ["Nominal CL", "Nominal AK", "Nominal TF", "Nominal TO",
                      "Total_Mutasi", "Selisih_Final", "Jumlah_Titik_Mutasi"]
    kolom_numerik = [c for c in kolom_numerik if c in audit_df.columns]
    audit_df[kolom_numerik] = audit_df[kolom_numerik].apply(pd.to_numeric, errors="coerce").fillna(0)

    jumlah_sebelum = len(audit_df)
    audit_df = audit_df.drop_duplicates().reset_index(drop=True)
    print(f"[handle_missing_and_duplicates] baris duplikat dihapus: {jumlah_sebelum - len(audit_df)}")

    return audit_df


def engineer_date_features(audit_df):
    """Ekstraksi fitur dari kolom tanggal."""
    audit_df = audit_df.copy()
    audit_df["Tanggal"] = pd.to_datetime(audit_df["Tanggal"], errors="coerce")
    audit_df["Hari"] = audit_df["Tanggal"].dt.day
    audit_df["Hari_dalam_Minggu"] = audit_df["Tanggal"].dt.dayofweek
    audit_df["Akhir_Pekan"] = audit_df["Hari_dalam_Minggu"].isin([5, 6]).astype(int)
    return audit_df


def detect_outliers(audit_df, kolom="Total_Mutasi"):
    """Deteksi outlier dengan metode IQR (ditandai, bukan dihapus)."""
    audit_df = audit_df.copy()
    Q1 = audit_df[kolom].quantile(0.25)
    Q3 = audit_df[kolom].quantile(0.75)
    IQR = Q3 - Q1
    batas_bawah = Q1 - 1.5 * IQR
    batas_atas = Q3 + 1.5 * IQR

    audit_df["Is_Outlier_Nominal"] = (~audit_df[kolom].between(batas_bawah, batas_atas)).astype(int)
    print(f"[detect_outliers] outlier terdeteksi: {audit_df['Is_Outlier_Nominal'].sum()} dari {len(audit_df)} baris")
    return audit_df


def encode_categorical(audit_df):
    """Label-encode kolom kategorikal. Mengembalikan (audit_df, encoders)."""
    audit_df = audit_df.copy()
    encoders = {}

    le_jenis_bayar = LabelEncoder()
    audit_df["Jenis_Bayar_Encoded"] = le_jenis_bayar.fit_transform(audit_df["Jenis Bayar"])
    encoders["Jenis Bayar"] = le_jenis_bayar

    if "Akun Closing" in audit_df.columns:
        le_akun_closing = LabelEncoder()
        audit_df["Akun_Closing_Encoded"] = le_akun_closing.fit_transform(audit_df["Akun Closing"])
        encoders["Akun Closing"] = le_akun_closing

    return audit_df, encoders


def scale_numeric(audit_df, kolom_numerik=None):
    """Standarisasi kolom numerik nominal. Mengembalikan (audit_df, scaler, kolom_hasil)."""
    audit_df = audit_df.copy()
    if kolom_numerik is None:
        kolom_numerik = ["Nominal CL", "Nominal AK", "Nominal TF", "Nominal TO",
                          "Total_Mutasi", "Selisih_Final"]
    kolom_numerik = [c for c in kolom_numerik if c in audit_df.columns]

    scaler = StandardScaler()
    fitur_scaled = pd.DataFrame(
        scaler.fit_transform(audit_df[kolom_numerik]),
        columns=[f"{c}_scaled" for c in kolom_numerik],
        index=audit_df.index,
    )
    audit_df = pd.concat([audit_df, fitur_scaled], axis=1)
    kolom_hasil = list(fitur_scaled.columns)
    return audit_df, scaler, kolom_hasil


def split_dataset(audit_df, kolom_scaled, test_size=0.2, random_state=42):
    """Menyusun fitur final dan melakukan train-test split."""
    fitur_model = list(kolom_scaled) + [
        "Jumlah_Titik_Mutasi", "Hari", "Hari_dalam_Minggu",
        "Akhir_Pekan", "Is_Outlier_Nominal", "Jenis_Bayar_Encoded",
    ]
    if "Akun_Closing_Encoded" in audit_df.columns:
        fitur_model.append("Akun_Closing_Encoded")

    X = audit_df[fitur_model]
    y = audit_df["Target_ML"]

    kelas_cukup_untuk_stratify = y.value_counts().min() >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if kelas_cukup_untuk_stratify else None,
    )

    print(f"[split_dataset] X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, fitur_model


# =====================================================================
# 6. SIMPAN OUTPUT
# =====================================================================
def save_outputs(output_dir, audit_df_final, X_train, X_test, y_train, y_test,
                  scaler, encoders, label_mapping):
    os.makedirs(output_dir, exist_ok=True)

    audit_df_final.to_csv(os.path.join(output_dir, "audit_df_preprocessed.csv"), index=False)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    for nama_kolom, encoder in encoders.items():
        nama_file = "label_encoder_" + re.sub(r"\W+", "_", nama_kolom.lower()) + ".joblib"
        joblib.dump(encoder, os.path.join(output_dir, nama_file))

    with open(os.path.join(output_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({int(k): v for k, v in enumerate(label_mapping)}, f, indent=2, ensure_ascii=False)

    print(f"[save_outputs] semua artefak tersimpan di: {output_dir}")
    for fname in sorted(os.listdir(output_dir)):
        print("  -", fname)


# =====================================================================
# 7. ORKESTRATOR UTAMA
# =====================================================================
def preprocess_data(
    raw_dir="namadataset_raw",
    output_dir="namadataset_preprocessing",
    laporan_kasir_file="Laporan kasir.xlsx.gpg",
    closing_file="data sistem 1 - 31 mei.xlsx.gpg",
    closing_sheet="PendapatanPerAkunClosing",
    aruskas_file="ARUS KAS 2026.xlsx.gpg",
    aruskas_sheet="Mei",
    test_size=0.2,
    random_state=42,
):
    """
    Fungsi utama: menjalankan seluruh pipeline preprocessing secara
    otomatis dari data mentah hingga dataset siap latih.

    Mengembalikan dict berisi X_train, X_test, y_train, y_test,
    audit_df_final, scaler, encoders, dan label_mapping.
    """
    # 1. Load
    df_to, df_tf, df_cl, df_ak = load_raw_data(
        raw_dir,
        laporan_kasir_file=laporan_kasir_file,
        closing_file=closing_file,
        closing_sheet=closing_sheet,
        aruskas_file=aruskas_file,
        aruskas_sheet=aruskas_sheet,
    )

    # 2. Bersihkan tiap sumber
    df_to = clean_df_to(df_to)
    df_tf = clean_df_tf(df_tf)
    df_cl = clean_df_cl(df_cl)
    df_ak = clean_df_ak(df_ak)

    # 3-4. Rekonsiliasi + pelabelan
    audit_df, label_mapping = build_audit_dataframe(df_to, df_tf, df_cl, df_ak)

    # 5. Finalisasi preprocessing untuk ML
    audit_df = handle_missing_and_duplicates(audit_df)
    audit_df = engineer_date_features(audit_df)
    audit_df = detect_outliers(audit_df)
    audit_df, encoders = encode_categorical(audit_df)
    audit_df, scaler, kolom_scaled = scale_numeric(audit_df)
    X_train, X_test, y_train, y_test, fitur_model = split_dataset(
        audit_df, kolom_scaled, test_size=test_size, random_state=random_state
    )

    # 6. Simpan
    save_outputs(output_dir, audit_df, X_train, X_test, y_train, y_test,
                 scaler, encoders, label_mapping)

    return {
        "audit_df_final": audit_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_columns": fitur_model,
        "scaler": scaler,
        "encoders": encoders,
        "label_mapping": label_mapping,
    }


# =====================================================================
# 8. ENTRY POINT CLI
# =====================================================================
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocessing otomatis data rekonsiliasi keuangan klinik untuk Machine Learning."
    )
    parser.add_argument("--raw_dir", default="keuangan_raw",
                         help="Folder berisi file Excel data mentah.")
    parser.add_argument("--output_dir", default="preprocessing/keuangan_preprocessing",
                         help="Folder tujuan menyimpan hasil preprocessing.")
    parser.add_argument("--laporan_kasir_file", default="Laporan kasir.xlsx.gpg")
    parser.add_argument("--closing_file", default="data sistem 1 - 31 mei.xlsx.gpg")
    parser.add_argument("--closing_sheet", default="PendapatanPerAkunClosing")
    parser.add_argument("--aruskas_file", default="ARUS KAS 2026.xlsx.gpg")
    parser.add_argument("--aruskas_sheet", default="Mei")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    preprocess_data(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        laporan_kasir_file=args.laporan_kasir_file,
        closing_file=args.closing_file,
        closing_sheet=args.closing_sheet,
        aruskas_file=args.aruskas_file,
        aruskas_sheet=args.aruskas_sheet,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print("\nPreprocessing selesai.")
