import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Pengaturan visualisasi
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
pd.set_option('display.max_columns', None)


def preprocess_heart_disease_pipeline(
    file_path,
    output_dir="heart_disease_pipeline",
    test_size=0.2,
    random_state=42,
    visualize=True,
    save_outputs=True
):
    """
    Pipeline preprocessing lengkap untuk dataset penyakit jantung.
    
    Parameters
    ----------
    file_path : str
        Path ke file CSV dataset
    output_dir : str
        Folder utama untuk menyimpan semua output
    test_size : float
        Proporsi data test (default 0.2)
    random_state : int
        Seed untuk reproducibility
    visualize : bool
        Apakah membuat visualisasi EDA
    save_outputs : bool
        Apakah menyimpan CSV, gambar, dan model objects
    
    Returns
    -------
    dict
        Berisi semua data dan objek penting untuk modeling & inference
    """
    
    print("="*70)
    print("HEART DISEASE PREPROCESSING PIPELINE DIMULAI".center(70))
    print("="*70)
    
    # ==================== 1. Buat Folder Output ====================
    folders = {
        'root': output_dir,
        'visualizations': os.path.join(output_dir, 'visualizations'),
        'csv': os.path.join(output_dir, 'csv_output'),
        'data': os.path.join(output_dir, 'data')
    }
    
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    # ==================== 2. Load Dataset ====================
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Dataset berhasil dimuat: {df.shape[0]} baris × {df.shape[1]} kolom")
    
    # ==================== 3. Rename Kolom ke Bahasa Indonesia ====================
    rename_map = {
        'id': 'id_pasien', 'age': 'usia', 'origin': 'asal_studi', 'sex': 'jenis_kelamin',
        'cp': 'tipe_nyeri_dada', 'trestbps': 'tekanan_darah_istirahat', 'chol': 'kolesterol',
        'fbs': 'gula_darah_puasa', 'restecg': 'hasil_ecg_istirahat', 'thalach': 'detak_jantung_maksimal',
        'exang': 'angina_olahraga', 'oldpeak': 'depresi_st', 'slope': 'kemiringan_st',
        'ca': 'jumlah_pembuluh_darah', 'thal': 'thalassemia', 'num': 'target'
    }
    df.rename(columns=rename_map, inplace=True)
    print("Nama kolom diubah ke Bahasa Indonesia")
    
    # ==================== 4. Copy untuk Processing ====================
    df_clean = df.copy()
    
    # ==================== 5. Handling Missing Values ====================
    print("\n--- Handling Missing Values ---")
    missing_before = df_clean.isnull().sum().sum()
    if missing_before > 0:
        num_cols = df_clean.select_dtypes(include=np.number).columns
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        
        for col in num_cols:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        for col in cat_cols:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        print(f"{missing_before} missing values telah diisi")
    else:
        print("Tidak ada missing values")
    
    # ==================== 6. Hapus Duplikat ====================
    dup_before = df_clean.duplicated().sum()
    if dup_before > 0:
        df_clean.drop_duplicates(inplace=True)
        print(f"{dup_before} baris duplikat dihapus")
    else:
        print("Tidak ada duplikat")
    
    # ==================== 7. Handling Outliers (IQR) ====================
    print("\n--- Handling Outliers (IQR) ---")
    def remove_outliers_iqr(df_temp, col):
        Q1 = df_temp[col].quantile(0.25)
        Q3 = df_temp[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((df_temp[col] < lower) | (df_temp[col] > upper)).sum()
        df_temp = df_temp[(df_temp[col] >= lower) & (df_temp[col] <= upper)]
        return df_temp, outliers
    
    outlier_cols = ['tekanan_darah_istirahat', 'kolesterol', 'detak_jantung_maksimal', 'depresi_st']
    total_outliers = 0
    initial_rows = len(df_clean)
    
    for col in outlier_cols:
        if col in df_clean.columns:
            df_clean, n_out = remove_outliers_iqr(df_clean, col)
            total_outliers += n_out
            if n_out > 0:
                print(f"   {col}: {n_out} outliers dihapus")
    
    print(f"Total data setelah outlier removal: {len(df_clean)} (dihapus {initial_rows - len(df_clean)} baris)")
    
    # ==================== 8. Encoding Kategorikal ====================
    print("\n--- Encoding Variabel Kategorikal ---")
    label_encoders = {}
    
    # Binary encoding
    if 'jenis_kelamin' in df_clean.columns:
        df_clean['jenis_kelamin'] = df_clean['jenis_kelamin'].map({
            'Male':1, 'M':1, 'male':1, 'Female':0, 'F':0, 'female':0
        })
        print("   jenis_kelamin → binary (1=Male, 0=Female)")
    
    boolean_cols = ['gula_darah_puasa', 'angina_olahraga']
    for col in boolean_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().map({
                '1':1, 'True':1, 'true':1, 'Yes':1, 'yes':1,
                '0':0, 'False':0, 'false':0, 'No':0, 'no':0
            }).fillna(0).astype(int)
            print(f"   {col} → binary")
    
    # Label encoding untuk ordinal
    ordinal_cols = ['tipe_nyeri_dada', 'hasil_ecg_istirahat', 'kemiringan_st', 'thalassemia']
    for col in ordinal_cols:
        if col in df_clean.columns:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            label_encoders[col] = le
            print(f"   {col} → label encoded")
    
    # Target: konversi ke binary (0 = tidak sakit, 1 = sakit)
    df_clean['target'] = (df_clean['target'] > 0).astype(int)
    print(f"Target dikonversi ke binary → distribusi: {dict(df_clean['target'].value_counts())}")
    
    # ==================== 9. Visualisasi (Opsional) ====================
    if visualize:
        print("\n--- Membuat Visualisasi EDA ---")
        
        # 1. Distribusi Target
        plt.figure(figsize=(14,5))
        plt.subplot(1,2,1)
        df_clean['target'].value_counts().plot(kind='bar', color=['#95e1d3', '#ff6b6b'])
        plt.title('Distribusi Target Penyakit Jantung')
        plt.xlabel('0 = Tidak Sakit, 1 = Sakit')
        plt.ylabel('Jumlah')
        
        plt.subplot(1,2,2)
        df_clean['target'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#95e1d3', '#ff6b6b'])
        plt.title('Proporsi Target')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(os.path.join(folders['visualizations'], '01_distribusi_target.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Distribusi Usia
        plt.figure()
        sns.histplot(data=df_clean, x='usia', hue='target', kde=True, bins=20, palette='Set1')
        plt.title('Distribusi Usia berdasarkan Penyakit Jantung')
        plt.xlabel('Usia (tahun)')
        plt.savefig(os.path.join(folders['visualizations'], '02_distribusi_usia.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Heatmap Korelasi
        plt.figure(figsize=(12,9))
        numeric_cols = df_clean.select_dtypes(include=np.number).columns
        if 'id_pasien' in numeric_cols:
            numeric_cols = numeric_cols.drop('id_pasien')
        corr = df_clean[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
        plt.title('Heatmap Korelasi Antar Variabel')
        plt.savefig(os.path.join(folders['visualizations'], '03_heatmap_korelasi.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   Semua visualisasi disimpan di folder visualizations/")
    
    # ==================== 10. Feature Scaling & Split ====================
    print("\n--- Feature Scaling & Train-Test Split ---")
    X = df_clean.drop(['target'], axis=1)
    if 'id_pasien' in X.columns:
        X = X.drop('id_pasien', axis=1)
    y = df_clean['target']
    
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Split selesai → Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # ==================== 11. Simpan Semua Output ====================
    if save_outputs:
        print("\n--- Menyimpan Output ---")
        
        # CSV
        df_clean.to_csv(os.path.join(folders['csv'], 'data_processed.csv'), index=False)
        pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(folders['csv'], 'train.csv'), index=False)
        pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(folders['csv'], 'test.csv'), index=False)
        
        # NPY
        np.save(os.path.join(folders['data'], 'X_train.npy'), X_train.values)
        np.save(os.path.join(folders['data'], 'X_test.npy'), X_test.values)
        np.save(os.path.join(folders['data'], 'y_train.npy'), y_train.values)
        np.save(os.path.join(folders['data'], 'y_test.npy'), y_test.values)
        
        # Pickle objects
        pipeline_objects = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'feature_names': X_train.columns.tolist(),
            'numeric_features': numeric_features,
            'df_raw': df,
            'df_processed': df_clean
        }
        
        with open(os.path.join(folders['data'], 'pipeline_objects.pkl'), 'wb') as f:
            pickle.dump(pipeline_objects, f)
        
        with open(os.path.join(folders['data'], 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        
        print("Semua file berhasil disimpan!")
    
    print("\n" + "="*70)
    print("PIPELINE SELESAI! DATA SIAP UNTUK MODELING".center(70))
    print("="*70)
    
    return pipeline_objects


# =============================================
# FUNGSI TAMBAHAN: PREDIKSI DATA BARU
# =============================================
def predict_new_data(pipeline_objects, new_data_df):
    """
    Fungsi untuk memproses data baru menggunakan pipeline yang sudah disimpan.
    """
    scaler = pipeline_objects['scaler']
    label_encoders = pipeline_objects['label_encoders']
    feature_names = pipeline_objects['feature_names']
    numeric_features = pipeline_objects['numeric_features']
    
    df = new_data_df.copy()
    
    # Rename kolom (sama seperti training)
    rename_map = {
        'id': 'id_pasien', 'age': 'usia', 'sex': 'jenis_kelamin', 'cp': 'tipe_nyeri_dada',
        'trestbps': 'tekanan_darah_istirahat', 'chol': 'kolesterol', 'fbs': 'gula_darah_puasa',
        'restecg': 'hasil_ecg_istirahat', 'thalach': 'detak_jantung_maksimal',
        'exang': 'angina_olahraga', 'oldpeak': 'depresi_st', 'slope': 'kemiringan_st',
        'ca': 'jumlah_pembuluh_darah', 'thal': 'thalassemia', 'num': 'target'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Encoding (sama seperti training)
    if 'jenis_kelamin' in df.columns:
        df['jenis_kelamin'] = df['jenis_kelamin'].map({'Male':1,'M':1,'Female':0,'F':0}).fillna(0)
    
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))
    
    # Drop kolom tidak terpakai
    if 'target' in df.columns:
        df = df.drop('target', axis=1)
    if 'id_pasien' in df.columns:
        df = df.drop('id_pasien', axis=1)
    
    # Scaling
    df[numeric_features] = scaler.transform(df[numeric_features])
    
    # Pastikan urutan kolom sama
    df = df[feature_names]
    
    return df


if __name__ == "__main__":
    hasil = preprocess_heart_disease_pipeline(
        file_path="heart_disease.csv",
        output_dir="heart_disease_pipeline",
        visualize=True,
        save_outputs=True
    )
    
    print("\nKunci yang tersedia:")
    for k in hasil.keys():
        print(f"   - {k}")
    
    print("\nPipeline siap digunakan untuk training model!")