# Proyek Akhir - Menyelesaikan Permasalahan Institusi Pendidikan: Prediksi Dropout Mahasiswa dan Visualisasi Performa Akademik di Jaya Jaya Institut

- [Proyek Akhir - Menyelesaikan Permasalahan Institusi Pendidikan: Prediksi Dropout Mahasiswa dan Visualisasi Performa Akademik di Jaya Jaya Institut](#proyek-akhir---menyelesaikan-permasalahan-institusi-pendidikan-prediksi-dropout-mahasiswa-dan-visualisasi-performa-akademik-di-jaya-jaya-institut)
  - [Business Understanding](#business-understanding)
    - [Latar Belakang](#latar-belakang)
    - [Permasalahan Bisnis](#permasalahan-bisnis)
    - [Cakupan Proyek](#cakupan-proyek)
    - [Persiapan](#persiapan)
      - [Sumber Data](#sumber-data)
      - [Setup Environment](#setup-environment)
  - [Business Dashboard](#business-dashboard)
  - [Sistem Machine Learning](#sistem-machine-learning)
    - [Studi Literatur](#studi-literatur)
    - [Evaluasi Model Machine Learning](#evaluasi-model-machine-learning)
    - [Menjelankan Inferensi Prototype Sistem Machine Learning di Streamlit](#menjelankan-inferensi-prototype-sistem-machine-learning-di-streamlit)
  - [Conclusion](#conclusion)
    - [Rekomendasi Action Items](#rekomendasi-action-items)

## Business Understanding

### Latar Belakang

Jaya Jaya Institut merupakan institusi pendidikan tinggi yang telah beroperasi sejak tahun 2000 dan berhasil mencetak banyak lulusan berkualitas. Meskipun demikian, terdapat tantangan serius yang dihadapi oleh institusi ini, yaitu tingginya tingkat mahasiswa yang tidak menyelesaikan studinya (dropout). Fenomena ini tidak hanya berdampak pada reputasi institusi, tetapi juga menyebabkan pemborosan sumber daya, baik dari sisi institusi maupun mahasiswa itu sendiri.

Tingginya angka dropout dapat disebabkan oleh berbagai faktor, mulai dari kondisi akademik, latar belakang sosial ekonomi, status keluarga, hingga motivasi pribadi mahasiswa. Sayangnya, identifikasi mahasiswa yang berisiko tinggi untuk dropout sering kali terlambat, sehingga institusi kehilangan kesempatan untuk memberikan intervensi yang tepat waktu.

Dengan kemajuan teknologi dan ketersediaan data, pendekatan berbasis data science dapat digunakan untuk mengatasi masalah ini. Melalui pemodelan prediktif dan analisis visualisasi, pihak institusi dapat mengenali pola-pola penting dari data mahasiswa dan memanfaatkan informasi tersebut untuk mendeteksi mahasiswa yang berpotensi dropout sejak dini. Dengan demikian, Jaya Jaya Institut dapat melakukan intervensi secara proaktif guna meningkatkan retensi dan keberhasilan akademik mahasiswa.

### Permasalahan Bisnis

Jaya Jaya Institut menghadapi tantangan serius berupa tingginya tingkat mahasiswa yang tidak menyelesaikan studinya (dropout). Fenomena ini memiliki dampak multi-dimensi yang merugikan institusi, antara lain:

1. Minimnya Sistem Peringatan Dini

   Tidak adanya sistem otomatis untuk mendeteksi potensi dropout sejak dini membuat intervensi yang diberikan cenderung bersifat reaktif. Institusi sering terlambat menyadari kondisi mahasiswa hingga mereka sudah benar-benar keluar atau menghilang dari perkuliahan.

2. Keterbatasan dalam Pemanfaatan Data Mahasiswa

   Walaupun Jaya Jaya Institut memiliki banyak data historis terkait mahasiswa—mulai dari latar belakang demografis, sosial-ekonomi, hingga performa akademik—data tersebut belum dimanfaatkan secara maksimal untuk pengambilan keputusan yang strategis.

3. Kurangnya Visualisasi dan Monitoring Kinerja Akademik

   Pihak manajemen kampus kesulitan memantau performa akademik mahasiswa secara agregat maupun individu karena belum tersedia dashboard interaktif yang menyajikan informasi dalam bentuk yang mudah dianalisis dan dipahami.

4. Kebutuhan Akan Alat Prediktif yang Terintegrasi

   Belum adanya sistem berbasis machine learning yang dapat digunakan secara langsung oleh staf akademik untuk memprediksi risiko dropout dan memberikan rekomendasi tindakan terhadap mahasiswa yang membutuhkan perhatian khusus.

Dengan adanya permasalahan-permasalahan tersebut, dibutuhkan solusi data-driven yang komprehensif: dari eksplorasi data, pemodelan prediksi dropout, visualisasi performa, hingga pembuatan sistem prototipe yang dapat digunakan secara langsung oleh institusi.

### Cakupan Proyek

Proyek ini bertujuan untuk membantu Jaya Jaya Institut dalam mengatasi permasalahan dropout dengan menerapkan pendekatan data science yang menyeluruh. Cakupan proyek meliputi:

1. Pemahaman Masalah dan Eksplorasi Data

   Mengkaji struktur dan karakteristik data mahasiswa yang mencakup faktor demografis, akademik, sosial ekonomi, dan performa perkuliahan untuk memahami penyebab utama dropout.

2. Pengembangan Model Prediktif

   Membangun dan mengevaluasi model machine learning untuk memprediksi kemungkinan dropout mahasiswa berdasarkan informasi historis dan kondisi awal saat pendaftaran.

3. Pembuatan Dashboard Interaktif

   Mengembangkan dashboard visualisasi yang informatif untuk membantu manajemen memantau tren dropout dan mengidentifikasi pola risiko secara real-time.

4. Implementasi Prototipe Aplikasi

   Menyediakan sistem prediksi berbasis Streamlit yang dapat digunakan oleh pihak kampus untuk memasukkan data mahasiswa baru dan mendapatkan estimasi risiko dropout secara langsung.

5. Rekomendasi Strategis

   Menyusun rekomendasi tindakan yang dapat diambil oleh institusi berdasarkan hasil analisis dan model prediksi untuk meningkatkan retensi mahasiswa.

Dengan cakupan ini, proyek diharapkan dapat memberikan solusi end-to-end yang tidak hanya menganalisis data historis, tetapi juga memberikan alat praktis bagi institusi untuk bertindak secara preventif dan strategis.

### Persiapan

#### Sumber Data

Dataset yang digunakan dalam proyek ini berasal dari UCI Machine Learning Repository dengan judul "Predict students' dropout and academic success". Dataset ini dapat diakses melalui tautan:

- https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance
- https://doi.org/10.24432/C5MC89

#### Setup Environment

```bash
s
```

## Business Dashboard

Jelaskan tentang business dashboard yang telah dibuat. Jika ada, sertakan juga link untuk mengakses dashboard tersebut.

## Sistem Machine Learning

### Studi Literatur

Dataset yang dipakai dalam proyek Student Dropout Prediction ini adalah salinan persis dari dataset yang dipublikasikan Realinho et al. (2022) pada makalah “Predicting Student Dropout and Academic Success” ([https://doi.org/10.3390/data7110146](https://doi.org/10.3390/data7110146)).

Realinho dkk. merilis open-data berisi 4442 mahasiswa × 35 atribut yang dikompilasi dari beragam basis data Politeknik Portalegre, Portugal. Atributnya meliputi demografi, sosial-ekonomi, makro-ekonomi, hingga performa akademik semester 1 dan 2. Masalah diformulasikan sebagai klasifikasi tiga kelas—Dropout, Enrolled, Graduate—dengan ketidakseimbangan label (50 % graduate, 32 % dropout, 18 % enrolled). Mereka mengeksplorasi algoritma RF, XGBoost, LightGBM, dan CatBoost, menilai kinerja via F1-score, serta menampilkan Permutation Feature Importance—fitur kinerja semester dan status pembayaran biaya kuliah muncul paling prognostik.

Artikel yang ditulis oleh Realinho dkk. dengan judul "Predicting Student Dropout and Academic Success" tahun 2022, datasetnya menjadi fondasi langsung bagi proyek Student Dropout Prediction untuk Jaya Jaya Institut.

1. Karakteristik Dataset
   - Ukuran & cakupan – 4 442 mahasiswa, 35 atribut yang menyeberang demografi, sosial-ekonomi, makro-ekonomi, data pendaftaran, serta performa akademik semester 1 dan 2 (periode 2008-2019).
   - Sumber – agregasi empat sistem internal & eksternal (Academic Management System, PAE, DGES, PORDATA) sehingga tidak ada nilai hilang.
   - Tugas – klasifikasi tiga kelas (Dropout, Enrolled, Graduate) pada akhir masa studi normal.
2. Permasalahan Ketidakseimbangan Kelas

   - Mayoritas Graduate 50 %, Dropout 32 %, Enrolled 18 %. Penulis menekankan perlunya penanganan imbalance melalui:
     - Level data : SMOTE, ADASYN dan variannya.
     - Level algoritma : Balanced Random Forest, Easy Ensemble, SMOTE-Bagging, dll.

3. Eksplorasi Data dan Insight Awal
   - Distribusi menurut program studi menunjukkan Nursing & Social Service paling sukses (≥70 % graduate) sementara Biofuel Tech & Informatics Engineering mencatat dropout tertinggi (≥54 %).
   - Faktor mahasiswa perempuan, penerima beasiswa, dan tuition fees up to date berkorelasi dengan kelulusan yang lebih baik.
   - Analisis multikolinearitas mengungkap korelasi kuat antar metrik semester (mis. approved S1 vs approved S2, r ≈ 0.90) serta Nationality ↔ International (r ≈ 0.91).
4. Metodologi Pemodelan

   Penulis menguji Random Forest (RF), XGBoost, LightGBM, dan CatBoost; evaluasi menggunakan macro-F1 untuk mengatasi bias akurasi pada data imbang-semu.

5. Temuan Fitur Penting

   Lima fitur konsisten penting di ke-4 model:

   | Rank | Fitur                                 | Keterangan         |
   | ---- | ------------------------------------- | ------------------ |
   | 1    | _Curricular units 2nd sem (approved)_ | Output semester 2  |
   | 2    | _Curricular units 1st sem (approved)_ | Output semester 1  |
   | 3    | _Curricular units 2nd sem (grade)_    | Rata-rata nilai S2 |
   | 4    | _Course_                              | Kode program studi |
   | 5    | _Tuition fees up to date_             | Status pembayaran  |

   Daftar tersebut muncul setelah uji Permutation Feature Importance pada semua algoritma.

6. Rekomendasi & Keterbatasan Paper
   - Fokus perbaikan: tangani imbalance, kurangi fitur redundant, perhatikan interpretabilitas model.
   - Batasan: data single-institution, belum ada integrasi sistem intervensi nyata; butuh pengayaan log LMS & alasan dropout

### Evaluasi Model Machine Learning

### Menjelankan Inferensi Prototype Sistem Machine Learning di Streamlit

Jelaskan cara menjalankan protoype sistem machine learning yang telah dibuat. Selain itu, sertakan juga link untuk mengakses prototype tersebut.

```bash

```

## Conclusion

Jelaskan konklusi dari proyek yang dikerjakan.

### Rekomendasi Action Items

Berikan beberapa rekomendasi action items yang harus dilakukan perusahaan guna menyelesaikan permasalahan atau mencapai target mereka.

- action item 1
- action item 2
