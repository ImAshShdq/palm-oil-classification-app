# Aplikasi Klasifikasi Kematangan Buah Kelapa Sawit

Aplikasi web cerdas untuk mengidentifikasi objek sebagai buah kelapa sawit dan mengklasifikasikan tingkat kematangannya (Matang, Mengkal, Mentah) berdasarkan analisis gambar.

## 🌟 Cara Kerja Aplikasi

Aplikasi ini menggunakan kombinasi analisis tekstur (GLCM), analisis distribusi warna (HSV), dan algoritma _Unsupervised Machine Learning_ (K-Means Clustering) untuk melakukan prediksi.

Arsitektur sistem dibagi menjadi dua tahap utama: **Validasi Objek** dan **Klasifikasi Kematangan**.

### 1. Ekstraksi Fitur (Feature Extraction)
Setiap gambar yang diunggah akan diekstrak 15 fiturnya:
*   **6 Fitur Tekstur GLCM (Gray Level Co-occurrence Matrix)**: Mengukur pola tekstur permukaan.
    *   *Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM*
*   **9 Fitur Warna HSV (Hue, Saturation, Value)**: Mengukur distribusi warna.
    *   *Mean, Standard Deviation, dan Skewness* untuk masing-masing channel H, S, dan V.

### 2. Validasi Objek (Dual-Gate Validation)
Sistem memiliki mekanisme pertahanan untuk mengenali apakah gambar yang diinput benar-benar kelapa sawit atau bukan (misalnya gambar batu, wajah, atau objek lain). Sistem menggunakan gerbang logika **AND** untuk 2 validasi:

1.  **Validasi Warna (HSV Z-Score):**
    *   Distribusi warna (9 fitur HSV) dari gambar input dibandingkan dengan rata-rata dan standar deviasi dari *seluruh* data latih kelapa sawit.
    *   Jika jarak _Z-Score_ melebihi ambang batas (Threshold ke-99 percentile) dari data latih, maka gambar ditolak karena **"Warna tidak cocok"** (contoh: warnanya ungu/biru padahal sawit harusnya dominan merah/oranye/hijau).
2.  **Validasi Tekstur (Jarak PCA K-Means):**
    *   Jika warna lolos, fitur gabungan (GLCM + HSV) direduksi dimensinya menjadi 2D menggunakan PCA (_Principal Component Analysis_).
    *   Titik 2D ini diukur jaraknya ke titik pusat (centroid) terdekat di model K-Means.
    *   Jika jaraknya melebihi ambang batas model, gambar ditolak karena **"Tekstur tidak cocok"**.

> Jika salah satu validasi gagal, objek dianggap **Bukan Kelapa Sawit**. Keduanya harus lolos agar masuk ke tahap klasifikasi.

### 3. Klasifikasi Kematangan (K-Means Clustering)
Jika gambar divalidasi sebagai kelapa sawit, model **K-Means (K=3)** akan menentukan tingkat kematangan berdasarkan klaster terdekat di ruang PCA. 

Klaster tidak ditentukan secara statis, melainkan diurutkan secara dinamis berdasarkan nilai **Mean Hue** (Rata-rata warna murni) dari tiap titik pusat klaster (centroid):
*   **Klaster 0 (Matang):** Memiliki nilai Hue terendah (dominan merah / oranye kemerahan).
*   **Klaster 1 (Mengkal):** Memiliki nilai Hue menengah (warna transisi).
*   **Klaster 2 (Mentah):** Memiliki nilai Hue tertinggi (dominan hijau gelap / kuning kehijauan).

---

## 🛠️ Struktur Proyek

```
palmoil-classification-app/
│
├── app.py                      # Flask Application Server (Routing & API)
├── requirements.txt            # Dependensi Python
├── README.md                   # Dokumentasi ini
│
├── dataset/                    # Folder Data Gambar
│   ├── train/                  # Data latih K-Means
│   └── test/                   # Data uji
│
├── ml_pipeline/                # Logika Machine Learning
│   ├── extract_features.py     # Wrapper ekstraksi GLCM & HSV
│   ├── extract_hsv_features.py # Modul khusus ekstraksi distribusi warna HSV (Mean, Std, Skew)
│   └── train_kmeans.py         # Script untuk melatih model, membuat threshold, & mapping label
│
├── models/                     # Model tersimpan (.pkl & .json) hasil training
│   ├── kmeans_model.pkl        
│   ├── pca_model.pkl
│   ├── scaler.pkl
│   ├── threshold.json          # Threshold validasi tekstur
│   ├── hsv_validation.json     # Threshold validasi distribusi warna
│   └── evaluation_metrics.json # Data dashboard (Silhouette score, scatter, dll)
│
├── static/                     # Aset Frontend
│   ├── style.css               # Styling UI (Card, Badges, HSV Channel colors)
│   └── script.js               # Logika UI (Upload, Render Metrics, Chart.js)
│
└── templates/                  # File HTML
    └── index.html              # Halaman utama aplikasi
```

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Prasyarat Sistem
Pastikan Python 3 sudah terinstal. Buka terminal dan masuk ke direktori proyek.

### 2. Install Dependensi
```bash
pip install -r requirements.txt
```

### 3. Latih Ulang Model (Opsional)
Jika Anda menambahkan gambar baru ke folder `dataset/train/`, latih ulang model dengan perintah:
```bash
python3 ml_pipeline/train_kmeans.py
```
Proses ini akan mengekstrak fitur, melatih K-Means, menghitung PCA, dan menghasilkan file JSON validasi (`hsv_validation.json` & `threshold.json`).

### 4. Jalankan Server Web
```bash
python3 app.py
```

### 5. Buka Aplikasi
Buka browser dan akses alamat berikut:
```
http://127.0.0.1:5000
```
Anda bisa langsung melakukan _Drag and Drop_ gambar untuk memprediksi kematangan sawit.

---

## 🔬 Metrik Tampilan Dasbor
Pada halaman utama bagian bawah, aplikasi menampilkan dasbor evaluasi model yang dibentuk dari hasil _training_:
- **Silhouette Score:** Mengukur seberapa padat dan terpisahnya klaster (mendekati 1 semakin baik).
- **Distribusi Data:** Grafik batang jumlah data latih dan prediksi data uji.
- **Visualisasi PCA (Live Scatter Plot):** Memetakan posisi sebaran gambar di ruang 2D. Gambar yang Anda unggah secara _real-time_ akan muncul sebagai ikon ⭐ di grafik ini.
