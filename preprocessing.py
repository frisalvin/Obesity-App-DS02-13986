import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import streamlit as st

def preprocess_data(df_input, show=True):
    # Buat salinan DataFrame untuk menghindari modifikasi DataFrame asli di luar fungsi
    df = df_input.copy()

    if show:
        st.subheader("🧹 Langkah-langkah Pra-pemrosesan Data")
        st.markdown("---") # Garis pemisah untuk keterbacaan

        st.markdown("### 🔄 Mengganti '?' dengan NaN")
        st.write("Data sebelum penggantian (contoh baris yang mengandung '?'):")
        # Tampilkan baris yang mengandung '?' sebelum diganti
        st.dataframe(df[df.astype(str).isin(['?']).any(axis=1)].head())

    # Ganti '?' dengan NaN
    df.replace('?', np.nan, inplace=True)

    # Definisikan kolom numerik. Pastikan ini sesuai dengan dataset Anda
    num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    
    # Konversi kolom numerik ke tipe numerik. 'errors='coerce' akan mengubah nilai non-numerik menjadi NaN
    # Ini sangat penting jika ada data kotor selain '?' yang mungkin ada di kolom numerik
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Hapus baris di mana kolom numerik yang kritis memiliki NaN setelah konversi
    # Ini penting untuk membersihkan entri yang sangat rusak
    initial_row_count = len(df)
    df.dropna(subset=num_cols, inplace=True)
    if show:
        rows_dropped_numeric_nan = initial_row_count - len(df)
        if rows_dropped_numeric_nan > 0:
            st.warning(f"Jumlah baris yang dihapus karena nilai non-numerik (menjadi NaN) di kolom numerik: **{rows_dropped_numeric_nan}**")
        else:
            st.info("Tidak ada baris yang dihapus karena nilai non-numerik di kolom numerik.")
        st.write("Contoh data setelah '?' diganti NaN dan konversi numerik:")
        st.dataframe(df.head())

        st.markdown("---")
        st.markdown("### 🔍 Missing Values Sebelum Imputasi")
        missing_before_impute = df.isnull().sum()
        missing_before_impute = missing_before_impute[missing_before_impute > 0]
        if not missing_before_impute.empty:
            st.dataframe(missing_before_impute)
        else:
            st.success("Tidak ada missing values yang terdeteksi sebelum imputasi (setelah konversi numerik dan penghapusan baris awal).")

    # Identifikasi ulang kolom kategorikal setelah konversi numerik
    # Ini memastikan kita hanya mengimputasi kolom 'object' yang tersisa
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    
    if show:
        st.markdown("---")
        st.markdown("### 🛠️ Imputasi Missing Values")
        st.markdown("#### Strategi Imputasi:")
        st.markdown("- **Kolom Kategorikal:** Diisi menggunakan **modus** (nilai paling sering muncul).")
        st.markdown("- **Kolom Numerik:** Diisi menggunakan **median** (nilai tengah).")

    # Imputasi missing values untuk kolom kategorikal (modus)
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            df[col] = imputer.fit_transform(df[[col]]).ravel()
            if show:
                st.write(f"📁 Kolom kategorikal '**{col}**' diimputasi dengan modus.")

    # Imputasi missing values untuk kolom numerik (median)
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            imputer = SimpleImputer(strategy='median')
            df[col] = imputer.fit_transform(df[[col]])
            if show:
                st.write(f"📊 Kolom numerik '**{col}**' diimputasi dengan median.")

    if show:
        st.markdown("### ✅ Contoh Data Setelah Imputasi")
        st.dataframe(df.head())

        st.markdown("### 🔍 Missing Values Setelah Imputasi")
        missing_after_impute = df.isnull().sum()
        missing_after_impute = missing_after_impute[missing_after_impute > 0]
        if not missing_after_impute.empty:
            st.error("⚠️ **Peringatan:** Masih ada missing values yang tersisa setelah imputasi!")
            st.dataframe(missing_after_impute)
            # Menghentikan eksekusi jika masih ada NaN, karena ini akan menyebabkan masalah di langkah selanjutnya
            raise ValueError("NaNs terdeteksi di data setelah imputasi. Mohon periksa kembali proses imputasi.")
        else:
            st.success("Tidak ada missing values setelah proses imputasi.")

    # Hapus duplikat
    if show:
        st.markdown("---")
        st.markdown("### 🧽 Pembersihan Duplikat")
    before_dup = len(df)
    df.drop_duplicates(inplace=True)
    after_dup = len(df)
    if show:
        st.write(f"Jumlah data sebelum hapus duplikat: **{before_dup}**")
        st.write(f"Jumlah data setelah hapus duplikat: **{after_dup}**")
        rows_dropped_duplicates = before_dup - after_dup
        if rows_dropped_duplicates > 0:
            st.warning(f"Jumlah data duplikat yang dihapus: **{rows_dropped_duplicates}**")
        else:
            st.info("Tidak ada data duplikat yang ditemukan dan dihapus.")

    # Deteksi dan penanganan outlier (menggunakan IQR)
    if show:
        st.markdown("---")
        st.markdown("### 🚨 Penanganan Outlier (IQR)")
        st.write("Visualisasi outlier pada kolom 'Age', 'Height', 'Weight' **Sebelum** penanganan:")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5)) # Perbesar ukuran figure
        for i, col in enumerate(['Age', 'Height', 'Weight']):
            sns.boxplot(y=df[col], ax=axes[i])
            axes[i].set_title(f"{col} (Sebelum)")
        plt.tight_layout() # Pastikan layout rapi
        st.pyplot(fig)

    original_len_after_dup = len(df)
    for col in ['Age', 'Height', 'Weight']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    if show:
        rows_dropped_outlier = original_len_after_dup - len(df)
        if rows_dropped_outlier > 0:
            st.warning(f"Jumlah baris yang dihapus karena outlier: **{rows_dropped_outlier}**")
        else:
            st.info("Tidak ada outlier yang signifikan ditemukan dan dihapus pada kolom yang dipilih.")

        st.write("Visualisasi outlier pada kolom 'Age', 'Height', 'Weight' **Sesudah** penanganan:")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, col in enumerate(['Age', 'Height', 'Weight']):
            sns.boxplot(y=df[col], ax=axes[i])
            axes[i].set_title(f"{col} (Sesudah)")
        plt.tight_layout()
        st.pyplot(fig)

    # Encoding fitur kategorikal
    if show:
        st.markdown("---")
        st.markdown("### 🔄 Encoding Fitur Kategorikal (Label Encoding)")

    # Perbarui daftar kolom kategorikal setelah penghapusan duplikat/outlier jika ada perubahan
    cat_cols_for_encoding = df.select_dtypes(include='object').columns.tolist()
    # Pastikan kolom target tidak ikut di-encode di sini jika sudah berupa 'object'
    if 'NObeyesdad' in cat_cols_for_encoding:
        cat_cols_for_encoding.remove('NObeyesdad')
    
    label_encoders = {}
    for col in cat_cols_for_encoding:
        le = LabelEncoder()
        # Verifikasi ulang tidak ada NaN sebelum encoding
        if df[col].isnull().any():
            st.error(f"Kolom '**{col}**' masih mengandung NaN sebelum Label Encoding!")
            raise ValueError(f"NaNs terdeteksi di kolom '{col}' sebelum Label Encoding.")
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        if show:
            st.write(f"🔧 Kolom '**{col}**' berhasil di-encode.")

    # Encoding kolom target 'NObeyesdad'
    target_encoder = LabelEncoder()
    if df['NObeyesdad'].isnull().any():
        st.error("Kolom target '**NObeyesdad**' masih mengandung NaN sebelum Label Encoding!")
        raise ValueError("NaNs terdeteksi di kolom target 'NObeyesdad'.")
    df['NObeyesdad'] = target_encoder.fit_transform(df['NObeyesdad'])
    if show:
        st.write("🎯 Kolom target '**NObeyesdad**' berhasil di-encode.")
        st.markdown("### 🔤 Contoh Data Setelah Encoding (Beberapa Baris Awal)")
        st.dataframe(df.head())

    # Pisahkan fitur (X) dan target (y)
    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']

    # Pengecekan NaN final sebelum Scaling dan SMOTE
    if X.isnull().sum().sum() > 0:
        st.error("⚠️ **Error Kritis:** NaN values terdeteksi di fitur (X) sebelum StandardScaler!")
        st.dataframe(X.isnull().sum()[X.isnull().sum() > 0])
        raise ValueError("NaN values terdeteksi di fitur (X) sebelum penskalaan. Periksa preprocessing Anda.")
    if y.isnull().sum() > 0:
        st.error("⚠️ **Error Kritis:** NaN values terdeteksi di target (y) sebelum SMOTE!")
        st.dataframe(y.isnull().sum())
        raise ValueError("NaN values terdeteksi di target (y) sebelum SMOTE. Periksa preprocessing Anda.")

    # Normalisasi fitur menggunakan StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    if show:
        st.markdown("---")
        st.markdown("### 📈 Normalisasi Fitur (StandardScaler)")
        st.write("Fitur numerik telah dinormalisasi menggunakan `StandardScaler`.")
        st.dataframe(X.head()) # Tampilkan beberapa baris pertama setelah scaling

    if show:
        st.markdown("---")
        st.markdown("### 🔥 Visualisasi Korelasi Antar Fitur")
        # Menghitung korelasi setelah semua transformasi numerik
        fig = plt.figure(figsize=(16, 12))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title("Heatmap Korelasi antar Fitur", fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        st.write("Heatmap ini menunjukkan hubungan antara setiap pasangan fitur. Nilai mendekati 1 atau -1 menunjukkan korelasi kuat.")

    if show:
        st.markdown("---")
        st.markdown("### 📊 Distribusi Kelas Target Sebelum dan Sesudah SMOTE")
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Sebelum SMOTE")
            fig = plt.figure(figsize=(5, 4))
            sns.countplot(x=y)
            plt.title("Distribusi Kelas Sebelum SMOTE")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

    # Penanganan ketidakseimbangan kelas dengan SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    if show:
        with col2:
            st.write("#### Setelah SMOTE")
            fig = plt.figure(figsize=(5, 4))
            sns.countplot(x=y_resampled)
            plt.title("Distribusi Kelas Setelah SMOTE")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
        st.write("SMOTE digunakan untuk menyamakan jumlah sampel di setiap kelas target, mengatasi masalah ketidakseimbangan data.")

    # Pembagian dataset menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
    
    if show:
        st.markdown("---")
        st.markdown("### 📦 Pembagian Dataset (Latih & Uji)")
        st.write(f"Ukuran data latih (fitur): **{X_train.shape}**")
        st.write(f"Ukuran data latih (target): **{y_train.shape}**")
        st.write(f"Ukuran data uji (fitur): **{X_test.shape}**")
        st.write(f"Ukuran data uji (target): **{y_test.shape}**")
        st.write("Dataset telah dibagi menjadi 80% untuk pelatihan model dan 20% untuk pengujian, dengan stratifikasi untuk menjaga proporsi kelas.")

        st.markdown("---")
        st.markdown("### 📝 Kesimpulan Pra-pemrosesan Data")
        st.markdown("""
        Proses pra-pemrosesan data ini telah menyiapkan dataset Anda untuk pelatihan model machine learning dengan serangkaian langkah yang komprehensif:
        - **Penanganan Missing Values:** Nilai '?' diganti dengan `NaN`, kemudian `NaN` pada kolom kategorikal diisi dengan **modus** dan pada kolom numerik dengan **median**. Baris yang memiliki nilai non-numerik yang sangat rusak juga telah dihapus.
        - **Pembersihan Duplikat:** Entri data yang duplikat telah berhasil dihapus untuk memastikan keunikan data.
        - **Penanganan Outlier:** Outlier pada kolom 'Age', 'Height', dan 'Weight' telah ditangani menggunakan metode **IQR** (Interquartile Range) untuk mengurangi pengaruh nilai ekstrem.
        - **Encoding Fitur Kategorikal:** Semua fitur kategorikal (selain kolom target) telah dikonversi menjadi representasi numerik menggunakan **Label Encoding**.
        - **Normalisasi Fitur Numerik:** Fitur-fitur numerik telah diskalakan menggunakan **StandardScaler** untuk menstandarisasi rentang nilai, yang penting untuk performa optimal banyak algoritma machine learning.
        - **Koreksi Ketidakseimbangan Kelas:** Distribusi kelas target yang tidak seimbang telah diperbaiki menggunakan **SMOTE (Synthetic Minority Over-sampling Technique)**, yang menghasilkan sampel sintetis untuk kelas minoritas.
        - **Pembagian Dataset:** Data akhir telah dibagi menjadi **data latih (80%)** dan **data uji (20%)** untuk evaluasi model yang adil. Pembagian ini juga menggunakan stratifikasi untuk memastikan proporsi kelas yang sama di set latih dan uji.
        """)

    return (X_train, X_test, y_train, y_test), label_encoders, target_encoder
