# Iris Species Prediction Dashboard 🌺

Sebuah aplikasi web interaktif, minimalis, dan mudah digunakan untuk memprediksi jenis bunga Iris (Setosa, Versicolor, atau Virginica) berdasarkan ukuran kelopak bunga secara real-time. 

Aplikasi ini dirancang khusus dengan antarmuka non-teknis agar mudah dipahami oleh siapa saja, serta telah dioptimalkan agar responsif pada perangkat seluler (mobile-friendly).

## 🚀 Fitur Utama
*   **Minimalist Blue Theme**: Desain antarmuka bersih, modern, dan berkontras tinggi untuk kenyamanan membaca.
*   **Pilihan Pengisian Cepat (Preset)**: Memuat contoh ukuran bunga khas Setosa, Versicolor, dan Virginica secara instan untuk pengujian cepat.
*   **Panduan Visual Bunga**: Gambar petunjuk letak kelopak luar (Sepal) dan kelopak dalam (Petal) langsung pada panel pengatur.
*   **Grafik Dimensi Kelopak**: Visualisasi diagram batang horizontal interaktif yang membandingkan dimensi kelopak bunga.
*   **Mobile Optimized**: Penyesuaian ukuran otomatis dan spanduk panduan navigasi khusus saat diakses melalui HP.

## 🛠️ Cara Menjalankan Aplikasi

1. **Persiapan Lingkungan (Conda)**
   Aktifkan environment Anda dan pastikan dependencies telah terinstal:
   ```bash
   conda activate project-streamlit
   ```

2. **Jalankan Server Streamlit**
   Jalankan perintah berikut di folder proyek:
   ```bash
   streamlit run app.py
   ```

3. **Akses Aplikasi**
   Buka browser Anda dan akses tautan lokal:
   ```text
   http://localhost:8501
   ```

## 📁 Struktur File Proyek
*   `app.py`: Kode utama aplikasi Streamlit.
*   `best_svm_model.pkl`: Model klasifikasi SVM yang telah dilatih dan dibungkus di dalam preprocessor pipeline (StandardScaler).
*   `iris_guide.png`: Gambar panduan bagian kelopak bunga Iris.
*   `Sesi 11 - Learn Studio.ipynb`: Jupyter notebook tempat pengolahan data, pembuatan pipeline, pencarian parameter terbaik, dan ekspor model.
*   `.streamlit/config.toml`: Konfigurasi server Streamlit untuk berjalan dalam mode headless tanpa telemetri.
