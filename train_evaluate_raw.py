import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def train_and_evaluate_raw(show=True):
    if show:
        st.header("💡 Pelatihan & Evaluasi Model (Tanpa Pre-processing dan Tuning)")

    df_raw = pd.read_csv("data/ObesityDataSet.csv")

    if show:
        st.markdown("### 📋 Data Asli (Belum Diolah)")
        # Menampilkan baris yang mengandung '?' atau NaN untuk inspeksi awal
        st.dataframe(df_raw[df_raw.isnull().any(axis=1) | df_raw.apply(lambda row: row.astype(str).str.contains(r'\?').any(), axis=1)].head())
    
    # --- Perbaikan: Pastikan kolom numerik dikonversi ke numerik sejak awal ---
    numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    for col in numerical_cols:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce') # Konversi ke numerik, ganti non-numerik jadi NaN
    
    # Ganti '?' dengan NaN dan hapus baris yang mengandung NaN
    df_raw.replace('?', np.nan, inplace=True) # Ini sudah ada, tapi konversi numerik harus duluan
    df_raw.dropna(inplace=True) # Hapus baris dengan NaN setelah konversi

    # Pisahkan fitur dan target
    X_raw = df_raw.drop('NObeyesdad', axis=1)
    y_raw = df_raw['NObeyesdad']

    # Encode variabel target
    target_encoder = LabelEncoder()
    y_raw_encoded = target_encoder.fit_transform(y_raw)

    # Encode fitur kategorikal di X_raw
    X_encoded = X_raw.copy()
    for col in X_encoded.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])

    # Pembagian dataset
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_encoded, y_raw_encoded, test_size=0.2, random_state=42)

    # Definisi model
    models_raw = {
        # --- Perbaikan: Tingkatkan max_iter untuk Logistic Regression ---
        "Logistic Regression (Raw)": LogisticRegression(max_iter=5000), # Ditingkatkan dari 1000
        "Random Forest (Raw)": RandomForestClassifier(random_state=42),
        "KNN (Raw)": KNeighborsClassifier()
    }

    results_raw = {}

    # Pelatihan dan evaluasi model
    for model_name, model in models_raw.items():
        if show:
            st.markdown(f"### 🚀 Model: {model_name}")
        start = time.time()
        model.fit(X_train_raw, y_train_raw)
        y_pred_raw = model.predict(X_test_raw)
        duration = time.time() - start

        # Hitung metrik evaluasi
        acc = accuracy_score(y_test_raw, y_pred_raw)
        prec = precision_score(y_test_raw, y_pred_raw, average='weighted', zero_division=0)
        rec = recall_score(y_test_raw, y_pred_raw, average='weighted')
        f1 = f1_score(y_test_raw, y_pred_raw, average='weighted')

        results_raw[model_name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "y_pred": y_pred_raw,
            "Training Time (s)": duration
        }

        if show:
            st.markdown(f"**Akurasi: {acc:.4f}**")
            st.text(classification_report(y_test_raw, y_pred_raw, target_names=target_encoder.classes_))
            st.write("**Confusion Matrix:**")
            cm = confusion_matrix(y_test_raw, y_pred_raw)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                        xticklabels=target_encoder.classes_,
                        yticklabels=target_encoder.classes_, ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)

    # Tampilkan hasil dalam DataFrame dan grafik
    results_df_raw = pd.DataFrame(results_raw).T[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Training Time (s)']]

    if show:
        st.markdown("### 📊 Tabel Evaluasi Model")
        st.dataframe(results_df_raw.style.format("{:.4f}"))
        st.markdown("### 📈 Grafik Perbandingan Performa")
        fig_metrics, ax_metrics = plt.subplots(figsize=(10, 6))
        results_df_raw[['Accuracy', 'Precision', 'Recall', 'F1 Score']].plot(kind='bar', ax=ax_metrics, colormap='Set2')
        ax_metrics.set_title("Perbandingan Performa Model (Tanpa Pre-processing & Tuning)")
        ax_metrics.set_ylabel("Skor")
        ax_metrics.set_ylim(0, 1.05)
        ax_metrics.set_xticklabels(results_df_raw.index, rotation=0)
        ax_metrics.grid(axis='y')
        st.pyplot(fig_metrics)
        st.markdown("### 📝 Kesimpulan Pelatihan Model Tanpa Pre-processing dan Tunning")
        st.markdown("""
        - Pelatihan model klasifikasi dengan Logistic Regression dan KNN tanpa adanya pre-processing menunjukkan hasil yang kurang optimal.
        - Dalam kasus Random Forest, algoritma ini menunjukkan performa yang sudah bagus sejak awal, dengan skor rata-rata di atas 90% bahkan tanpa pre-processing dan tuning.
        - Dari sini terlihat bahwa pre-processing dapat secara signifikan membantu meningkatkan kinerja model, terutama untuk Logistic Regression dan KNN.
        """)

    return results_df_raw