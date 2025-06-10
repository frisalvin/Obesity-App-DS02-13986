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
    for col in num_cols:
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

    initial_rows_after_impute_dup = len(df)
    for col in ['Age', 'Height', 'Weight']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    if show:
        st.write(f"Jumlah baris setelah penanganan outlier: {len(df)} (Dihapus: {initial_rows_after_impute_dup - len(df)})")

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

    # --- START OF MODIFIED/ADDITIONS FOR ROBUSTNESS ---
    if show:
        st.markdown("### 🚧 Final Data Validation Before Scaling and SMOTE 🚧")

    # Pastikan X dan y bebas dari NaN dan inf sebelum konversi ke NumPy array
    # Imputasi NaN/inf di X sebelum konversi
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            if (X[col].isnull().any() or X[col].isin([np.inf, -np.inf]).any()):
                if show:
                    st.write(f"🔧 Mengatasi NaN/inf di kolom X '{col}' sebelum konversi ke NumPy array.")
                X[col].replace([np.inf, -np.inf], np.nan, inplace=True)
                # Gunakan median untuk numerik
                X[col] = X[col].fillna(X[col].median())
    
    # Imputasi NaN/inf di y sebelum konversi
    if pd.api.types.is_numeric_dtype(y):
        if (y.isnull().any() or y.isin([np.inf, -np.inf]).any()):
            if show:
                st.write("🔧 Mengatasi NaN/inf di target y sebelum konversi ke NumPy array.")
            y.replace([np.inf, -np.inf], np.nan, inplace=True)
            # Untuk target, modus mungkin lebih aman karena ini adalah label kategorikal yang sudah di-encode
            y = y.fillna(y.mode()[0])

    # Konversi X ke numpy array (float64 adalah default yang aman untuk fitur numerik)
    X_np = X.to_numpy(dtype=np.float64) 
    
    # Konversi y ke numpy array. Pastikan sudah bersih dari NaN/inf karena dtype=np.int64 tidak bisa ada NaN.
    y_np = y.to_numpy(dtype=np.int64) 

    # Final Check for NaNs/Infs (should ideally not happen here if previous steps are correct)
    if np.any(np.isnan(X_np)) or np.any(np.isinf(X_np)):
        if show:
            st.error("❌ ERROR FATAL: NaN atau inf masih terdeteksi di X_np setelah penanganan! Periksa logika.")
            st.dataframe(pd.DataFrame(X_np).isnull().sum()[pd.DataFrame(X_np).isnull().sum() > 0])
            st.dataframe(pd.DataFrame(X_np).isin([np.inf, -np.inf]).sum())
        raise ValueError("Critical NaNs or infinite values still present in X_np before scaling.")

    if np.any(np.isnan(y_np)) or np.any(np.isinf(y_np)):
        if show:
            st.error("❌ ERROR FATAL: NaN atau inf masih terdeteksi di y_np setelah penanganan! Periksa logika.")
            st.write(np.isnan(y_np).sum())
            st.write(np.isinf(y_np).sum())
        raise ValueError("Critical NaNs or infinite values still present in y_np before scaling.")

    # Konversi kembali ke DataFrame/Series untuk Standard Scaler
    X = pd.DataFrame(X_np, columns=X.columns)
    y = pd.Series(y_np, name='NObeyesdad')

    # Standard Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns) # X after scaling

    # --- END OF MODIFIED/ADDITIONS FOR ROBUSTNESS ---

    if show:
        st.markdown("### 🔥 Korelasi antar Fitur")
        # Pastikan df yang digunakan untuk korelasi sudah bersih dan numerik
        df_for_corr_viz = X.copy()
        df_for_corr_viz['NObeyesdad'] = y 
        
        fig = plt.figure(figsize=(16, 12))
        sns.heatmap(df_for_corr_viz.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, cbar_kws={"shrink": 0.8})
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
    # Gunakan X dan y yang sudah divalidasi dan discale
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
