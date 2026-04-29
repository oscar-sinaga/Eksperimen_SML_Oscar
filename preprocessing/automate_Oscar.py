import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)

def preprocess_data(df, scaler_save_path):
    print("Memulai proses preprocessing tingkat lanjut...")
    df_clean = df.copy()

    # 1. Menghapus Kolom yang Tidak Relevan (Mencegah Data Leakage)
    print(" -> Menghapus kolom UDI, Product ID, dan Failure Type...")
    df_clean = df_clean.drop(['UDI', 'Product ID', 'Failure Type'], axis=1, errors='ignore')

    # 2. FEATURE ENGINEERING (Menciptakan Sinyal Mekanis)
    print(" -> Melakukan Feature Engineering (Temp_Difference, Power, Strain)...")
    df_clean['Temp_Difference'] = df_clean['Process temperature [K]'] - df_clean['Air temperature [K]']
    df_clean['Power'] = df_clean['Torque [Nm]'] * df_clean['Rotational speed [rpm]']
    df_clean['Strain'] = df_clean['Tool wear [min]'] * df_clean['Rotational speed [rpm]']

    # 3. FEATURE SELECTION (Mencegah Multikolinearitas)
    print(" -> Menghapus suhu asli untuk mencegah tumpang tindih informasi...")
    df_clean = df_clean.drop(['Air temperature [K]', 'Process temperature [K]'], axis=1, errors='ignore')

    # 4. ENCODING KATEGORIKAL (Ordinal Encoding untuk Kualitas)
    print(" -> Melakukan Ordinal Encoding pada kolom Type...")
    type_mapping = {'L': 0, 'M': 1, 'H': 2}
    if 'Type' in df_clean.columns:
        df_clean['Type'] = df_clean['Type'].map(type_mapping)

    # 5. TRANSFORMASI LOGARITMIK (Penanganan Skewness Berdasarkan EDA)
    print(" -> Mengaplikasikan Log Transform pada fitur yang condong (skewed)...")
    skewed_cols = ['Rotational speed [rpm]', 'Tool wear [min]', 'Power', 'Strain']
    for col in skewed_cols:
        if col in df_clean.columns:
            df_clean[col] = np.log1p(df_clean[col])

    # 6. PEMISAHAN FITUR (X) DAN TARGET (y)
    X = df_clean.drop('Target', axis=1)
    y = df_clean['Target']

    # Train test split dilakukan di modelling untuk menjaga konsistensi dengan pipeline modelling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 7. SCALING (Standardisasi Tahan Outlier)
    print(" -> Melakukan Scaling dengan RobustScaler...")
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Menyimpan Scaler untuk Inference/Monitoring
    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    joblib.dump(scaler, scaler_save_path)
    print(f" -> [AMANKAN] Objek Scaler berhasil disimpan di {scaler_save_path}")

    # 8. MENGGABUNGKAN KEMBALI
    df_final_train = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
    df_final_test = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1)
    print("Preprocessing komprehensif selesai.")
    return df_final_train, df_final_test

def save_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data siap latih tersimpan di {output_path}")

if __name__ == "__main__":
    # Tentukan path secara dinamis
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "predictive_maintenance_raw", "predictive_maintenance.csv")
    CLEAN_DATA_PATH_TRAIN = os.path.join(BASE_DIR, "predictive_maintenance_preprocessing", "predictive_maintenance_train_processed.csv")
    CLEAN_DATA_PATH_TEST = os.path.join(BASE_DIR, "predictive_maintenance_preprocessing", "predictive_maintenance_test_processed.csv")
    SCALER_PATH = os.path.join(BASE_DIR, "predictive_maintenance_preprocessing", "scaler.pkl")

    # Eksekusi Pipeline
    try:
        raw_df = load_data(RAW_DATA_PATH)
        clean_df_train, clean_df_test = preprocess_data(raw_df, SCALER_PATH)
        save_data(clean_df_train, CLEAN_DATA_PATH_TRAIN)
        save_data(clean_df_test, CLEAN_DATA_PATH_TEST)
        print("=== PIPELINE PREPROCESSING BERHASIL DIEKSEKUSI ===")
    except Exception as e:
        print(f"=== TERJADI KESALAHAN ===\n{e}")