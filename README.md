# 🛡️ Sistem Deteksi Penipuan Transaksi (Fraud Detection Dashboard)

Sebuah *dashboard* simulasi interaktif berbasis Streamlit untuk mendeteksi penipuan (*fraud*) pada transaksi kartu kredit secara *real-time*.

Aplikasi ini menggunakan model **Support Vector Classifier (SVC)** yang telah dioptimasi menggunakan metode **SMOTE** (Synthetic Minority Over-sampling Technique) untuk menyeimbangkan data, serta **Genetic Algorithm (GA)** untuk seleksi fitur guna meningkatkan performa deteksi terhadap kasus penipuan minoritas.

## ✨ Fitur Utama
- **Simulasi Real-time:** Menyimulasikan transaksi masuk satu per satu atau secara otomatis.
- **Visualisasi Status Instan:** Alert warna yang responsif (hijau untuk aman, merah untuk *fraud* terdeteksi).
- **Pemrosesan Data Presisi:** Otomatis menyesuaikan data *input* (*scaling* dan urutan kolom) agar 100% cocok dengan struktur model saat *training*.
- **Tabel Riwayat Transaksi:** Melacak status transaksi sebelumnya secara berkelanjutan.
- **KPI Metrics:** Pemantauan statistik total transaksi diproses dan tingkat *fraud*.

## 🚀 Cara Menjalankan Aplikasi

### 1. Persiapan Dataset
Dataset yang digunakan berasal dari Kaggle (*Credit Card Fraud Detection*).
Karena ukurannya yang cukup besar (sekitar 150MB), pastikan Anda telah mengunduh dataset tersebut dan menempatkan file `creditcard.csv` di dalam *root directory* proyek (sejajar dengan file `app.py`).

### 2. Instalasi Dependensi
Sangat disarankan untuk menggunakan *virtual environment*. Instal semua *library* yang dibutuhkan dengan perintah berikut:

```bash
pip install -r requirements.txt
```

### 3. Menjalankan Dashboard
Buka terminal/CMD, pastikan Anda berada di direktori proyek ini, lalu jalankan:

```bash
streamlit run app.py
```

*Browser* akan otomatis terbuka pada alamat `http://localhost:8501` yang menampilkan *dashboard* simulasi Anda.

## 📂 Struktur Folder
- `app.py`: Kode utama aplikasi Streamlit.
- `creditcard.csv`: Dataset transaksi asli (tidak disertakan di Git).
- `requirements.txt`: Daftar dependensi Python.
- `model/`: Folder yang berisi *weights* model (`best_model.pkl`), *scalers* (`scaler_amount.pkl`, `scaler_time.pkl`), serta metadata urutan fitur hasil Genetic Algorithm (`feature_names.pkl`, `selected_features.pkl`).

## 🛠️ Catatan Teknis
Pastikan versi `scikit-learn` pada *environment* Anda cocok dengan yang ada pada `requirements.txt` untuk menghindari peringatan `InconsistentVersionWarning` dari file `.pkl` yang di-*load* (model dilatih menggunakan `scikit-learn` v1.6.1 atau di atasnya).
