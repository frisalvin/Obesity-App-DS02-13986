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
        # Tampilkan hanya jika ada '?'
        if df.isin(['?']).any(axis=1).any():
            st.dataframe(df[df.isin(['?']).any(axis=1)].head())
        else:
            st.info("Tidak ada nilai '?' yang ditemukan di data.")

    # Ganti '?' dengan NaN
    df.replace('?', np.nan, inplace=True)

    # Identifikasi kolom numerik dan kategorikal
    num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    
    # Konversi kolom numerik ke tipe numerik. 'errors='coerce' akan mengubah nilai non-numerik menjadi NaN
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # --- Tambahan: Hapus baris yang mungkin jadi NaN setelah pd.to_numeric ---
    initial_rows_after_q_replace = len(df)
    df.dropna(subset=num_cols, inplace=True) 
    if show and len(df) < initial_rows_after_q_replace:
        st.write(f"Dihapus {initial_rows_after_q_replace - len(df)} baris karena NaN pada kolom numerik setelah konversi.")


    if show:
        st.write("Contoh data setelah '?' diganti NaN dan konversi numerik:")
        st.dataframe(df.head())

        st.markdown("### 🔍 Missing Values Sebelum Imputasi")
        missing_before_impute = df.isnull().sum()
        missing_before_impute = missing_before_impute[missing_before_impute > 0]
        if not missing_before_impute.empty:
            st.dataframe(missing_before_impute)
        else:
            st.info("Tidak ada missing values sebelum imputasi.")


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
        missing_after_impute = df.isnull().sum()
        missing_after_impute = missing_after_impute[missing_after_impute > 0]
        if not missing_after_impute.empty:
            st.dataframe(missing_after_impute)
        else:
            st.info("Tidak ada missing values setelah imputasi.")


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

    # --- START OF CRITICAL FINAL DATA VALIDATION AND TYPE CONVERSION ---
    if show:
        st.markdown("### 🚧 Validasi Data Akhir Sebelum Scaling & SMOTE 🚧")

    # 1. Pastikan X bebas dari NaN dan inf
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            if X[col].isnull().any() or X[col].isin([np.inf, -np.inf]).any():
                if show:
                    st.warning(f"⚠️ NaN atau inf ditemukan di kolom X '{col}'. Mengimputasi kembali.")
                X[col].replace([np.inf, -np.inf], np.nan, inplace=True)
                imputer_final_X = SimpleImputer(strategy='median')
                X[col] = imputer_final_X.fit_transform(X[[col]]).ravel()
        elif not pd.api.types.is_numeric_dtype(X[col]): # Check if any non-numeric remains
            if show:
                st.error(f"❌ ERROR: Kolom '{col}' di X bukan tipe numerik setelah semua preprocessing.")
            raise TypeError(f"Kolom '{col}' di X bukan tipe numerik.")

    # 2. Pastikan y bebas dari NaN dan inf
    if y.isnull().any() or y.isin([np.inf, -np.inf]).any():
        if show:
            st.warning("⚠️ NaN atau inf ditemukan di target y. Mengimputasi kembali.")
        y.replace([np.inf, -np.inf], np.nan, inplace=True)
        imputer_final_y = SimpleImputer(strategy='most_frequent')
        y = imputer_final_y.fit_transform(y.to_frame()).ravel()
        y = pd.Series(y, name='NObeyesdad') 
    elif not pd.api.types.is_numeric_dtype(y): # Check if y is non-numeric
        if show:
            st.error("❌ ERROR: Target y bukan tipe numerik setelah semua preprocessing.")
        raise TypeError("Target y bukan tipe numerik.")

    # Konversi X dan y ke tipe data final yang non-nullable
    try:
        X_final = X.astype(np.float64) 
        y_final = y.astype(np.int64)   
    except Exception as e:
        if show:
            st.error(f"❌ ERROR: Gagal mengkonversi X atau y ke tipe data final (float64/int64). Detail: {e}")
            st.write("X dtypes sebelum konversi final:", X.dtypes)
            st.write("y dtype sebelum konversi final:", y.dtype)
            st.write("X nulls sebelum konversi final:", X.isnull().sum().sum())
            st.write("y nulls sebelum konversi final:", y.isnull().sum())
        raise ValueError(f"Final type conversion failed: {e}")

    # Final check sebelum scaling dan SMOTE
    if X_final.isnull().sum().sum() > 0:
        if show: st.error("❌ FATAL ERROR: NaN ditemukan di X_final sebelum scaling.")
        raise ValueError("NaNs found in X_final before scaling.")
    if y_final.isnull().sum() > 0:
        if show: st.error("❌ FATAL ERROR: NaN ditemukan di y_final sebelum scaling.")
        raise ValueError("NaNs found in y_final before scaling.")
    if not all(pd.api.types.is_numeric_dtype(X_final[col]) for col in X_final.columns):
        if show: st.error("❌ FATAL ERROR: Kolom non-numerik ditemukan di X_final sebelum scaling.")
        raise ValueError("Non-numeric columns found in X_final before scaling.")
    if not pd.api.types.is_numeric_dtype(y_final):
        if show: st.error("❌ FATAL ERROR: Target y_final bukan numerik sebelum scaling.")
        raise ValueError("Non-numeric target y_final found before scaling.")

    # Standard Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final) 
    X_processed = pd.DataFrame(X_scaled, columns=X_final.columns)

    # --- START OF NEW CRITICAL FINAL CONVERSION TO NUMPY ARRAYS FOR SMOTE ---
    if show:
        st.markdown("### 🔍 Final Konversi ke NumPy Array untuk SMOTE")

    # Konversi X_processed ke numpy array float64
    try:
        X_smote = X_processed.to_numpy(dtype=np.float64, copy=True)
    except Exception as e:
        if show:
            st.error(f"❌ ERROR: Gagal mengkonversi X_processed ke numpy.ndarray (float64) untuk SMOTE. Detail: {e}")
            st.write("X_processed dtypes:", X_processed.dtypes)
            st.write("X_processed nulls:", X_processed.isnull().sum().sum())
            st.write("X_processed infs:", X_processed.isin([np.inf, -np.inf]).sum().sum())
        raise ValueError(f"Final conversion of X for SMOTE failed: {e}")

    # Konversi y_final ke numpy array int64
    try:
        y_smote = y_final.to_numpy(dtype=np.int64, copy=True)
    except Exception as e:
        if show:
            st.error(f"❌ ERROR: Gagal mengkonversi y_final ke numpy.ndarray (int64) untuk SMOTE. Detail: {e}")
            st.write("y_final dtype:", y_final.dtype)
            st.write("y_final nulls:", y_final.isnull().sum())
            st.write("y_final infs:", y_final.isin([np.inf, -np.inf]).sum())
        raise ValueError(f"Final conversion of y for SMOTE failed: {e}")

    # Final check just before SMOTE:
    if np.any(np.isnan(X_smote)) or np.any(np.isinf(X_smote)):
        if show: st.error("❌ FATAL ERROR: NaN atau Inf ditemukan di X_smote tepat sebelum SMOTE.")
        raise ValueError("NaNs or Infs found in X_smote immediately before SMOTE.")
    if np.any(np.isnan(y_smote)) or np.any(np.isinf(y_smote)):
        if show: st.error("❌ FATAL ERROR: NaN atau Inf ditemukan di y_smote tepat sebelum SMOTE.")
        raise ValueError("NaNs or Infs found in y_smote immediately before SMOTE.")

    # --- END OF NEW CRITICAL FINAL CONVERSION TO NUMPY ARRAYS ---


    if show:
        st.markdown("### 🔥 Korelasi antar Fitur")
        df_for_corr_viz = X_processed.copy()
        df_for_corr_viz['NObeyesdad'] = y_final 
        
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
            sns.countplot(x=y_final) 
            plt.title("Distribusi Kelas Sebelum SMOTE")
            st.pyplot(fig)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_smote, y_smote)

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
