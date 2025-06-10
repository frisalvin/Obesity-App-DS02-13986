import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import streamlit as st

def preprocess_data(df, show=True):
    if show:
        st.subheader("🧹 Pre-processing Data")
        st.markdown("### 🔄 Mengganti '?' dengan NaN")
        st.write("Contoh data sebelum diganti:")
        st.dataframe(df[df.isin(['?']).any(axis=1)].head())

    # Ganti '?' dengan NaN
    df.replace('?', np.nan, inplace=True)

    # Identifikasi kolom numerik dan kategorikal
    num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    
    # Konversi kolom numerik ke tipe numerik. 'errors='coerce' akan mengubah nilai non-numerik menjadi NaN
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # --- Tambahan: Hapus baris yang mungkin jadi NaN setelah pd.to_numeric ---
    # Ini penting jika ada data kotor selain '?' yang berubah jadi NaN setelah konversi
    df.dropna(subset=num_cols, inplace=True) # Hapus baris di mana kolom numerik menjadi NaN

    if show:
        st.write("Contoh data setelah '?' diganti NaN dan konversi numerik:")
        # Tampilkan beberapa baris pertama setelah konversi dan dropna untuk inspeksi
        st.dataframe(df.head())

        st.markdown("### 🔍 Missing Values Sebelum Imputasi")
        st.dataframe(df.isnull().sum()[df.isnull().sum() > 0])

    # Kategorikan kolom setelah konversi numerik
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if show:
        st.markdown("### 🛠️ Imputasi Missing Values")
        st.markdown("#### Kategori → diisi pakai modus.")
        st.markdown("#### Numerik → diisi pakai median.")

    # Imputasi missing values untuk kolom kategorikal (modus)
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            df[col] = imputer.fit_transform(df[[col]]).ravel()
            if show:
                st.write(f"📁 Kolom kategori '{col}' diimputasi dengan modus.")

    # Imputasi missing values untuk kolom numerik (median)
    for col in num_cols: # Pastikan ini tetap dilakukan jika ada NaN dari sumber lain (misal, dari categorical setelah di-encode)
        if df[col].isnull().sum() > 0:
            imputer = SimpleImputer(strategy='median')
            df[col] = imputer.fit_transform(df[[col]])
            if show:
                st.write(f"📊 Kolom numerik '{col}' diimputasi dengan median.")

    if show:
        st.markdown("### ✅ Contoh Data Setelah Imputasi")
        st.dataframe(df.head())

        st.markdown("### 🔍 Missing Values Setelah Imputasi")
        st.dataframe(df.isnull().sum())

    # Hapus duplikat
    before_dup = len(df)
    df.drop_duplicates(inplace=True)
    after_dup = len(df)
    if show:
        st.markdown("### 🧽 Pembersihan Duplikat")
        st.write(f"Jumlah data sebelum hapus duplikat: {before_dup}")
        st.write(f"Jumlah data setelah hapus duplikat: {after_dup}")
        st.write(f"Jumlah data duplikat yang dihapus: {before_dup - after_dup}")

    # Deteksi dan penanganan outlier (menggunakan IQR)
    if show:
        st.markdown("### 🚨 Visualisasi Outlier Sebelum dan Sesudah")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, col in enumerate(['Age', 'Height', 'Weight']):
            sns.boxplot(y=df[col], ax=axes[i])
            axes[i].set_title(f"{col} (Sebelum)")
        st.pyplot(fig)

    for col in ['Age', 'Height', 'Weight']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    if show:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, col in enumerate(['Age', 'Height', 'Weight']):
            sns.boxplot(y=df[col], ax=axes[i])
            axes[i].set_title(f"{col} (Sesudah)")
        st.pyplot(fig)

    # Encoding fitur kategorikal
    if show:
        st.markdown("### 🔄 Encoding Fitur Kategorikal")

    cat_cols_after_impute = df.select_dtypes(include='object').columns.tolist()
    if 'NObeyesdad' in cat_cols_after_impute:
        cat_cols_after_impute.remove('NObeyesdad')
    
    label_encoders = {}
    for col in cat_cols_after_impute:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        if show:
            st.write(f"🔧 Kolom '{col}' berhasil di-encode.")

    target_encoder = LabelEncoder()
    df['NObeyesdad'] = target_encoder.fit_transform(df['NObeyesdad'])
    if show:
        st.write("🎯 Kolom target 'NObeyesdad' berhasil di-encode.")
        st.markdown("### 🔤 Contoh Data Setelah Encoding")
        st.dataframe(df.head())

    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    if show:
        st.markdown("### 🔥 Korelasi antar Fitur")
        fig = plt.figure(figsize=(16, 12))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title("Heatmap Korelasi antar Fitur", fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)

    if show:
        st.markdown("### 📊 Distribusi Kelas Sebelum dan Sesudah SMOTE")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Sebelum SMOTE")
            fig = plt.figure(figsize=(5, 4))
            sns.countplot(x=y)
            plt.title("Distribusi Kelas Sebelum SMOTE")
            st.pyplot(fig)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    if show:
        with col2:
            st.write("Setelah SMOTE")
            fig = plt.figure(figsize=(5, 4))
            sns.countplot(x=y_resampled)
            plt.title("Distribusi Kelas Setelah SMOTE")
            st.pyplot(fig)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    if show:
        st.markdown("### 📦 Pembagian Dataset")
        st.write(f"Ukuran data latih: {X_train.shape}")
        st.write(f"Ukuran data uji: {X_test.shape}")

        st.markdown("### 📝 Kesimpulan Preprocessing Data")
        st.markdown("""
        - Missing values berhasil diatasi dengan mengganti nilai '?' pada kolom kategorikal dengan **modus** dan pada kolom numerik dengan **median**.
        - Meskipun jumlah data kosong sedikit, informasi ini tetap penting karena dataset relatif kecil.
        - Duplikat berhasil dihapus, menyisakan data asli.
        - Semua fitur kategorikal telah dikonversi menjadi numerik menggunakan **Label Encoding**.
        - Tidak ada fitur yang memiliki korelasi sangat rendah terhadap target, sehingga **seluruh fitur tetap digunakan**.
        - Fitur numerik telah dinormalisasi menggunakan **StandardScaler** agar berada dalam skala seragam.
        - Distribusi kelas target yang tidak seimbang telah diperbaiki menggunakan **SMOTE**.
        - Dataset telah dibagi menjadi **data latih** dan **data uji** untuk keperluan pelatihan model, dengan dataset dibagi menjadi 80% data latih dan 20% data uji dengan 16 kolom fitur yang ada di dalam dataset.
        """)

    return (X_train, X_test, y_train, y_test), label_encoders, target_encoder