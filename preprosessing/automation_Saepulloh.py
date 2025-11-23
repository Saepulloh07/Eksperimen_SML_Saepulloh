import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')


class WinePreprocessor:
    """
    Preprocessor untuk Wine Quality Dataset dengan automation pipeline
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        
        # Pastikan folder output ada
        os.makedirs("data/output", exist_ok=True)
        print("[INFO] ✅ Preprocessor initialized")

    def load_dataset(self, url):
        """Load dataset dari URL"""
        print("[INFO] 📥 Loading dataset...")
        try:
            df = pd.read_csv(url, sep=';')
            print(f"[INFO] ✅ Dataset loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"[ERROR] ❌ Failed to load dataset: {e}")
            raise

    def feature_engineering(self, df):
        """Buat fitur baru"""
        print("[INFO] 🔧 Creating new features...")
        
        # Binary classification
        df['quality_label'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0)
        
        # Feature engineering
        df['total_acidity'] = df['fixed acidity'] + df['volatile acidity']
        df['acid_to_alcohol'] = df['total_acidity'] / df['alcohol']
        
        print("[INFO] ✅ Features created:")
        print("   - quality_label (binary: 0=bad, 1=good)")
        print("   - total_acidity")
        print("   - acid_to_alcohol")
        
        return df

    def remove_outliers(self, df):
        """Hapus outliers menggunakan IQR method"""
        print("[INFO] 🧹 Removing outliers...")
        original_count = df.shape[0]
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns 
                          if col not in ['quality', 'quality_label']]
        
        def iqr_filter(data, col):
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            return data[(data[col] >= lower) & (data[col] <= upper)]
        
        for col in numeric_columns:
            df = iqr_filter(df, col)
        
        removed = original_count - df.shape[0]
        print(f"[INFO] ✅ Outliers removed: {removed} samples ({removed/original_count*100:.2f}%)")
        
        return df

    def split_target(self, df):
        """Pisahkan features dan target"""
        print("[INFO] 🔀 Splitting features and target...")
        
        X = df.drop(['quality', 'quality_label'], axis=1)
        y = df['quality_label']
        
        self.feature_names = X.columns.tolist()
        
        print(f"[INFO] ✅ Split completed:")
        print(f"   - Features: {X.shape}")
        print(f"   - Target: {y.shape}")
        print(f"   - Feature count: {len(self.feature_names)}")
        
        return X, y

    def split_data(self, X, y):
        """Train-test split dengan stratification"""
        print("[INFO] 📊 Creating train-test split...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
        
        print(f"[INFO] ✅ Split completed:")
        print(f"   - Train set: {X_train.shape[0]} samples")
        print(f"   - Test set: {X_test.shape[0]} samples")
        print(f"   - Train distribution: {dict(y_train.value_counts())}")
        print(f"   - Test distribution: {dict(y_test.value_counts())}")
        
        return X_train, X_test, y_train, y_test

    def scale_data(self, X_train, X_test):
        """Standardize features"""
        print("[INFO] 🎯 Scaling features with StandardScaler...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"[INFO] ✅ Scaling completed:")
        print(f"   - Mean: {X_train_scaled.mean():.6f}")
        print(f"   - Std: {X_train_scaled.std():.6f}")
        
        return X_train_scaled, X_test_scaled

    def save_outputs(self, X_train, X_test, y_train, y_test):
        """Simpan semua output preprocessing"""
        print("[INFO] 💾 Saving preprocessed data...")
        
        try:
            # Save numpy arrays
            np.save("data/output/X_train_scaled.npy", X_train)
            np.save("data/output/X_test_scaled.npy", X_test)
            np.save("data/output/y_train.npy", y_train)
            np.save("data/output/y_test.npy", y_test)
            
            # Save scaler
            with open("data/output/scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
            
            # Save feature names
            with open("data/output/feature_names.pkl", "wb") as f:
                pickle.dump(self.feature_names, f)
            
            print("[INFO] ✅ All files saved successfully:")
            print("   - X_train_scaled.npy")
            print("   - X_test_scaled.npy")
            print("   - y_train.npy")
            print("   - y_test.npy")
            print("   - scaler.pkl")
            print("   - feature_names.pkl")
            
        except Exception as e:
            print(f"[ERROR] ❌ Failed to save files: {e}")
            raise

    def run_pipeline(self, url):
        """
        Jalankan full preprocessing pipeline
        """
        print("=" * 70)
        print("🚀 WINE QUALITY PREPROCESSING PIPELINE")
        print("=" * 70)
        
        try:
            # Step 1: Load
            df = self.load_dataset(url)
            
            # Step 2: Feature Engineering
            df = self.feature_engineering(df)
            
            # Step 3: Remove Outliers
            df = self.remove_outliers(df)
            
            # Step 4: Split Features & Target
            X, y = self.split_target(df)
            
            # Step 5: Train-Test Split
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            
            # Step 6: Scaling
            X_train_scaled, X_test_scaled = self.scale_data(X_train, X_test)
            
            # Step 7: Save
            self.save_outputs(X_train_scaled, X_test_scaled, y_train, y_test)
            
            print("\n" + "=" * 70)
            print("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            print(f"\n[ERROR] ❌ Pipeline failed: {e}")
            raise


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    # URL dataset
    DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    # Initialize & run
    processor = WinePreprocessor()
    X_train, X_test, y_train, y_test = processor.run_pipeline(DATA_URL)
    
    print(f"\n📊 Final Data Summary:")
    print(f"   - Training samples: {X_train.shape[0]}")
    print(f"   - Test samples: {X_test.shape[0]}")
    print(f"   - Features: {X_train.shape[1]}")
    print(f"\n✅ Ready for model training!")