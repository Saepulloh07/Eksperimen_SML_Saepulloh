import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')


class WinePreprocessor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None

        # Ensure output folder exists
        os.makedirs("data/output", exist_ok=True)

    # ======================================================
    # 1. LOAD DATA
    # ======================================================
    def load_dataset(self, url):
        print("[INFO] Loading dataset...")
        df = pd.read_csv(url, sep=';')
        print(f"[INFO] Dataset loaded. Shape = {df.shape}")
        return df

    # ======================================================
    # 2. FEATURE ENGINEERING
    # ======================================================
    def feature_engineering(self, df):
        print("[INFO] Applying feature engineering...")
        df['quality_label'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0)
        df['total_acidity'] = df['fixed acidity'] + df['volatile acidity']
        df['acid_to_alcohol'] = df['total_acidity'] / df['alcohol']
        print("[INFO] New features created: total_acidity, acid_to_alcohol")
        return df

    # ======================================================
    # 3. OUTLIER REMOVAL
    # ======================================================
    def remove_outliers(self, df):
        print("[INFO] Removing outliers...")
        original_count = df.shape[0]

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['quality', 'quality_label']]

        def iqr_filter(data, col):
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            return data[(data[col] >= lower) & (data[col] <= upper)]

        for col in numeric_columns:
            df = iqr_filter(df, col)

        print(f"[INFO] Outliers removed: {original_count - df.shape[0]} samples")
        return df

    # ======================================================
    # 4. SPLIT FEATURES & TARGET
    # ======================================================
    def split_target(self, df):
        print("[INFO] Splitting features and target...")
        X = df.drop(['quality', 'quality_label'], axis=1)
        y = df['quality_label']

        self.feature_names = X.columns.tolist()
        print(f"[INFO] Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y

    # ======================================================
    # 5. TRAIN-TEST SPLIT
    # ======================================================
    def split_data(self, X, y):
        print("[INFO] Splitting train/test data...")
        return train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    # ======================================================
    # 6. SCALING
    # ======================================================
    def scale_data(self, X_train, X_test):
        print("[INFO] Scaling features menggunakan StandardScaler...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    # ======================================================
    # 7. SAVE OUTPUT
    # ======================================================
    def save_outputs(self, X_train, X_test, y_train, y_test):
        print("[INFO] Dataset disimpan di data/output/...")

        np.save("data/output/X_train_scaled.npy", X_train)
        np.save("data/output/X_test_scaled.npy", X_test)
        np.save("data/output/y_train.npy", y_train)
        np.save("data/output/y_test.npy", y_test)

        with open("data/output/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        with open("data/output/feature_names.pkl", "wb") as f:
            pickle.dump(self.feature_names, f)

        print("[INFO] File berhasil disimpan!")

    # ======================================================
    # MASTER PIPELINE
    # ======================================================
    def load_and_preprocess(self, url):
        print("=" * 60)
        print(" Memulai pemprosesan Automation ")
        print("=" * 60)

        df = self.load_dataset(url)
        df = self.feature_engineering(df)
        df = self.remove_outliers(df)
        X, y = self.split_target(df)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        X_train_scaled, X_test_scaled = self.scale_data(X_train, X_test)
        self.save_outputs(X_train_scaled, X_test_scaled, y_train, y_test)

        print("\n[INFO] Pipeline berhasil dibuat!")
        print("=" * 60)

        return X_train_scaled, X_test_scaled, y_train, y_test


# ======================================================
# RUN PIPELINE
# ======================================================
if __name__ == "__main__":
    DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

    processor = WinePreprocessor()
    processor.load_and_preprocess(DATA_URL)
