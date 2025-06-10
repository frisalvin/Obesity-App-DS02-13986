import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV

# Fungsi dummy preprocess_data: Mengolah data mentah menjadi siap pakai
def preprocess_data(df, show=False):
    df_processed = df.copy()
    df_processed.replace('?', np.nan, inplace=True)
    
    # Konversi kolom numerik ke tipe numerik. 'errors='coerce' akan mengubah nilai non-numerik menjadi NaN
    numerical_cols_pre = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    for col in numerical_cols_pre:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
    df_processed.dropna(inplace=True) # Hapus baris dengan NaN setelah konversi

    X = df_processed.drop("NObeyesdad", axis=1)
    y = df_processed["NObeyesdad"]

    # Encode variabel target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    # Encode fitur kategorikal
    label_encoders = {}
    categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Skalakan fitur numerik
    scaler = StandardScaler()
    X[numerical_cols_pre] = scaler.fit_transform(X[numerical_cols_pre])

    return (X, X, y_encoded, y_encoded), label_encoders, target_encoder

# Fungsi utama aplikasi Streamlit
def run_text_classification():
    st.subheader("📥 Obesitas Klasifikasi")
    st.markdown("Silakan masukkan nilai dari setiap fitur berikut:")

    # Formulir input pengguna
    with st.form("manual_input_form"):
        Gender = st.selectbox("Gender (Jenis Kelamin)", options=["Male", "Female"])
        Age = st.number_input("Age (Usia, dalam tahun)", min_value=0, max_value=120, value=23)
        Height = st.number_input("Height (Tinggi badan dalam meter)", min_value=0.5, max_value=2.5, value=1.70, step=0.01)
        Weight = st.number_input("Weight (Berat badan dalam kg)", min_value=10.0, max_value=300.0, value=80.0, step=0.5)
        family_history = st.selectbox("Family History with Overweight (Riwayat keluarga kelebihan berat badan)", options=["yes", "no"])
        FAVC = st.selectbox("FAVC (Sering mengonsumsi makanan tinggi kalori)", options=["yes", "no"])
        FCVC = st.number_input("FCVC (Frekuensi konsumsi sayuran, 1-3)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
        NCP = st.number_input("NCP (Jumlah makanan utama per hari)", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
        CAEC = st.selectbox("CAEC (Konsumsi makanan di antara waktu makan)", options=["no", "Sometimes", "Frequently", "Always"])
        SMOKE = st.selectbox("SMOKE (Merokok)", options=["yes", "no"])
        CH2O = st.number_input("CH2O (Konsumsi air harian)", min_value=0.0, max_value=3.0, value=2.0, step=0.1)
        SCC = st.selectbox("SCC (Apakah mengamati konsumsi kalori sendiri)", options=["yes", "no"])
        FAF = st.number_input("FAF (Frekuensi aktivitas fisik mingguan)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
        TUE = st.number_input("TUE (Waktu penggunaan perangkat elektronik harian)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
        CALC = st.selectbox("CALC (Konsumsi alkohol)", options=["no", "Sometimes", "Frequently", "Always"])
        MTRANS = st.selectbox("MTRANS (Transportasi utama)", options=[
            "Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])

        submitted = st.form_submit_button("🔍 Klasifikasikan")

    if submitted:
        # Kumpulkan inputan pengguna ke dalam DataFrame
        input_dict = {
            'Gender': [Gender], 'Age': [Age], 'Height': [Height], 'Weight': [Weight],
            'family_history_with_overweight': [family_history], 'FAVC': [FAVC],
            'FCVC': [FCVC], 'NCP': [NCP], 'CAEC': [CAEC], 'SMOKE': [SMOKE],
            'CH2O': [CH2O], 'SCC': [SCC], 'FAF': [FAF], 'TUE': [TUE],
            'CALC': [CALC], 'MTRANS': [MTRANS]
        }
        df_input_original = pd.DataFrame(input_dict)

        # --- Perbaikan: Pastikan kolom numerik di df_input_original adalah numerik ---
        numerical_cols_input = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        for col in numerical_cols_input:
            # Konversi secara eksplisit. st.number_input sudah menjamin numerik, ini hanya pencegahan.
            df_input_original[col] = pd.to_numeric(df_input_original[col])
        
        st.markdown("### 📊 Inputan yang Digunakan untuk Prediksi")
        st.dataframe(df_input_original) # Ini tempat error PyArrow muncul jika tipe data salah

        # --- MODEL MENTAH (RAW MODEL) ---
        st.markdown("## 🔷 Hasil Model Tanpa Pre-processing & Tuning")
        df_raw = pd.read_csv("data/ObesityDataSet.csv")
        df_raw.replace('?', np.nan, inplace=True)
        
        # --- Perbaikan: Pastikan kolom numerik di df_raw dikonversi juga ---
        for col in numerical_cols_input: # Gunakan list kolom numerik yang sama
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
        
        df_raw.dropna(inplace=True)

        # Pisahkan fitur dan target untuk model mentah
        X_raw = df_raw.drop("NObeyesdad", axis=1)
        y_raw = df_raw["NObeyesdad"]
        target_encoder_raw = LabelEncoder().fit(y_raw)
        y_raw_enc = target_encoder_raw.transform(y_raw)

        df_input_raw_model = df_input_original.copy() # Salinan input untuk model mentah

        # Encode fitur kategorikal untuk model mentah dan input pengguna
        categorical_cols_raw = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
        for col in categorical_cols_raw:
            le = LabelEncoder()
            X_raw[col] = le.fit_transform(X_raw[col]) # Fit dan transform pada data pelatihan
            
            # Transform input pengguna menggunakan encoder yang sama
            val = str(df_input_raw_model[col][0])
            if val not in le.classes_:
                st.error(f"❌ Nilai '{val}' tidak valid untuk kolom '{col}' (Raw Model).")
                return
            df_input_raw_model[col] = le.transform([val])

        # Pastikan urutan fitur input cocok dengan data pelatihan
        feature_order_raw = X_raw.columns.tolist()
        df_input_raw_model = df_input_raw_model[feature_order_raw]

        # Latih dan prediksi dengan model mentah
        models_raw = {
            # --- Perbaikan: Tingkatkan max_iter untuk Logistic Regression ---
            "Logistic Regression (Raw)": LogisticRegression(max_iter=5000), # Ditingkatkan
            # --- Perbaikan: Tambahkan random_state untuk reproduktifitas ---
            "Random Forest (Raw)": RandomForestClassifier(random_state=42),
            "KNN (Raw)": KNeighborsClassifier()
        }

        for name, model in models_raw.items():
            model.fit(X_raw, y_raw_enc)
            pred = model.predict(df_input_raw_model)
            st.write(f"**{name}**: {target_encoder_raw.inverse_transform(pred)[0]}")

        # --- MODEL SETELAH PRA-PEMROSESAN ---
        st.markdown("## 🔷 Hasil Model Setelah Pre-processing")
        df_full_data = pd.read_csv("data/ObesityDataSet.csv")
        # Pra-pemrosesan data menggunakan fungsi preprocess_data
        (X_train, _, y_train, _), label_encoders_preprocessed, target_encoder_preprocessed = preprocess_data(df_full_data, show=False)

        df_input_preprocessed_model = df_input_original.copy() # Salinan input untuk model yang diproses

        # Terapkan encoding menggunakan encoder yang difit dari pra-pemrosesan data pelatihan
        categorical_cols_preprocessed = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
        for col in categorical_cols_preprocessed:
            le = label_encoders_preprocessed[col]
            val = str(df_input_preprocessed_model[col][0])
            if val not in le.classes_:
                st.error(f"❌ Nilai '{val}' tidak valid untuk kolom '{col}' (Preprocessed Model).")
                return
            df_input_preprocessed_model[col] = le.transform([val])

        # Terapkan penskalaan menggunakan scaler yang difit dari data pelatihan asli
        numerical_cols_preprocessed_inference = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        
        # Buat scaler baru dan fit pada numerical_cols dari data asli (sebelum diproses)
        # Ini penting agar scaler yang digunakan untuk input baru sama dengan scaler yang digunakan
        # untuk data pelatihan X_train (yang diskalakan oleh preprocess_data)
        temp_df_for_scaler_fit = df_full_data.copy()
        temp_df_for_scaler_fit.replace('?', np.nan, inplace=True)
        # --- Perbaikan: Pastikan kolom numerik di temp_df_for_scaler_fit juga dikonversi ---
        for col in numerical_cols_preprocessed_inference:
            temp_df_for_scaler_fit[col] = pd.to_numeric(temp_df_for_scaler_fit[col], errors='coerce')
        temp_df_for_scaler_fit.dropna(inplace=True)
        
        scaler_for_inference = StandardScaler()
        scaler_for_inference.fit(temp_df_for_scaler_fit[numerical_cols_preprocessed_inference])

        df_input_preprocessed_model[numerical_cols_preprocessed_inference] = scaler_for_inference.transform(df_input_preprocessed_model[numerical_cols_preprocessed_inference])

        # Pastikan urutan fitur cocok dengan X_train
        feature_order_preprocessed = X_train.columns.tolist()
        df_input_preprocessed_model = df_input_preprocessed_model[feature_order_preprocessed]

        # Latih dan prediksi dengan model yang sudah diproses
        models_pre = {
            # --- Perbaikan: Tingkatkan max_iter untuk Logistic Regression ---
            "Logistic Regression (Preprocessed)": LogisticRegression(max_iter=5000), # Ditingkatkan
            # --- Perbaikan: Tambahkan random_state untuk reproduktifitas ---
            "Random Forest (Preprocessed)": RandomForestClassifier(random_state=42),
            "KNN (Preprocessed)": KNeighborsClassifier()
        }

        for name, model in models_pre.items():
            model.fit(X_train, y_train)
            pred = model.predict(df_input_preprocessed_model)
            st.write(f"**{name}**: {target_encoder_preprocessed.inverse_transform(pred)[0]}")

        # --- MODEL SETELAH TUNING ---
        st.markdown("## 🔷 Hasil Model Setelah Tuning")
        # Definisi parameter grid untuk tuning hyperparameter
        param_grid = {
            'Logistic Regression': {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']},
            'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
            'KNN': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
        }

        # Model dasar sebelum tuning
        base_models = {
            # --- Perbaikan: Tingkatkan max_iter untuk Logistic Regression ---
            'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42), # Ditingkatkan
            'Random Forest': RandomForestClassifier(random_state=42),
            'KNN': KNeighborsClassifier()
        }

        # Lakukan tuning dan prediksi dengan model terbaik
        for name in base_models:
            grid = GridSearchCV(base_models[name], param_grid[name], cv=3, scoring='f1_weighted', n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_ # Dapatkan model terbaik setelah tuning
            
            pred = best_model.predict(df_input_preprocessed_model)
            st.write(f"**{name} (Tuned)**: {target_encoder_preprocessed.inverse_transform(pred)[0]}")