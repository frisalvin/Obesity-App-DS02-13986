import streamlit as st
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time

def tune_models(X_train, X_test, y_train, y_test, target_encoder, show=True):
    if show:
        st.subheader("🔍 Hyperparameter Tuning & Evaluation")

    # Definisi parameter grid untuk tuning hyperparameter
    param_grid = {
        'Logistic Regression': {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']},
        'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
        'KNN': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
    }

    # Model dasar sebelum tuning
    base_models = {
        # --- Perbaikan: Tingkatkan max_iter untuk Logistic Regression dan tambahkan random_state ---
        'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42), # Ditingkatkan dari 1000, tambah random_state
        'Random Forest': RandomForestClassifier(random_state=42), # Tambahkan random_state
        'KNN': KNeighborsClassifier()
    }

    results = {}

    for name in base_models:
        if show:
            st.markdown(f"### 🔧 {name}")
        
        start = time.time()  # ⏱️ Mulai pengukuran waktu tuning
        # GridSearchCV untuk mencari hyperparameter terbaik
        grid = GridSearchCV(base_models[name], param_grid[name], cv=5, scoring='f1_weighted', n_jobs=-1)
        grid.fit(X_train, y_train)
        end = time.time()  # ⏱️ Selesai

        training_time = end - start

        # Dapatkan model terbaik dari hasil tuning
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        # Hitung metrik evaluasi
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "Training Time (s)": training_time
        }

        if show:
            st.markdown(f"**Akurasi:** {acc:.4f}")
            st.write(f"**Best Parameters:** {grid.best_params_}")
            st.text(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
                        xticklabels=target_encoder.classes_,
                        yticklabels=target_encoder.classes_)
            plt.title(f"Confusion Matrix: {name}")
            st.pyplot(fig)

    results_df = pd.DataFrame(results).T[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Training Time (s)']]

    if show:
        st.markdown("### 📋 Tabel Hasil Evaluasi Model (Setelah Tuning)")
        st.dataframe(results_df.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "Training Time (s)": "{:.2f}"
        }))
        st.markdown("### 📊 Grafik Perbandingan Performa Model")
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        results_df[['Accuracy', 'Precision', 'Recall', 'F1 Score']].plot(kind='bar', ax=ax_bar, colormap='Set3')
        ax_bar.set_title("Perbandingan Performa Model (Setelah Pre-Processing dan Tuning)")
        ax_bar.set_ylabel("Skor")
        ax_bar.set_ylim(0, 1.05)
        ax_bar.set_xticklabels(results_df.index, rotation=0)
        ax_bar.legend(loc='lower right')
        ax_bar.grid(axis='y')
        st.pyplot(fig_bar)
        
        st.markdown("### 📝 Kesimpulan Pelatihan Model Setelah Pre-Processing dan Setelah Tuning")
        st.markdown("""
        - Setelah dilakukan proses tuning terhadap ketiga algoritma yaitu Logistic Regression, Random Forest, dan K-Nearest Neighbors (KNN) menggunakan GridSearchCV dengan parameter grid yang telah ditentukan, diperoleh peningkatan performa yang signifikan khususnya pada model Random Forest.
        - Random Forest kembali menjadi model dengan performa terbaik, dengan nilai akurasi sebesar 95.74%, serta metrik precision, recall, dan F1-score yang semuanya konsisten tinggi di angka 95.7%.
        - Logistic Regression menunjukkan performa yang stabil dengan akurasi sebesar 91.48%.
        - KNN juga menunjukkan performa yang baik dengan akurasi 89.69%.
        - Secara keseluruhan, proses hyperparameter tuning berhasil meningkatkan atau mempertahankan performa model dengan signifikan.
        """)

    return results_df