# 🛡️ Big Data Security Analytics Dashboard

**UAS Perancangan Big Data**  
Sistem Deteksi Anomali & Monitoring Keamanan Data Berbasis Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🔗 Live Dashboard

> **Akses Dashboard**: [https://uas-big-data.streamlit.app](https://uas-big-data.streamlit.app)  
> *(Setelah deployment di Streamlit Community Cloud)*

---

## 📋 Deskripsi Proyek

Dashboard ini menganalisis **5.000 event keamanan** dari sistem enterprise untuk mendeteksi anomali dan potensi ancaman keamanan data. Menggunakan **IsolationForest** (unsupervised machine learning) untuk mengidentifikasi pola perilaku mencurigakan pada user.

### Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| 📊 **Visualisasi Interaktif** | Histogram, Bar Chart, Pie Chart, dan Line Chart |
| 🤖 **Deteksi Anomali** | IsolationForest dengan evaluasi Precision, Recall, F1-Score |
| 🚨 **Real-Time Alert** | Simulasi log alert secara dinamis (LOW → CRITICAL) |
| 🔍 **Data Explorer** | Jelajahi data events, users, dan fitur engineering |
| 🔒 **Mitigasi Risiko** | Deteksi otomatis akses dari user terminated |

---

## 🏗️ Arsitektur & Struktur Kode

```
UAS-BIG-DATA/
├── app.py                      # Dashboard utama Streamlit
├── model.py                    # IsolationForest + evaluasi model
├── utils.py                    # Data cleaning & feature engineering
├── sample_stream_events.csv    # Dataset events (5.000 records)
├── users.csv                   # Dataset users (150 records)
├── assets.csv                  # Dataset assets (8 records)
├── stream_generator.py         # Generator data streaming
├── requirements.txt            # Dependencies Python
├── .gitignore                  # Git ignore rules
└── README.md                   # Dokumentasi (file ini)
```

### Penjelasan Modul

#### `utils.py` — Pembersihan Data & Feature Engineering
- `load_data()` — Memuat CSV dari path relatif
- `clean_events()` — Konversi tipe, hapus duplikat, isi missing values
- `compute_user_features()` — Menghitung 8 fitur keamanan per user:
  - `failed_login_rate` — Rasio login gagal (indikator brute-force)
  - `avg_latency` — Rata-rata waktu respons (deteksi bot)
  - `access_to_restricted_ratio` — Rasio akses data sensitif
  - `total_bytes_out` — Volume data keluar (deteksi exfiltration)
  - `avg_risk_score`, `unique_assets`, `unique_ips`, dll.
- `security_alert()` — Klasifikasi event ke LOW/MEDIUM/HIGH/CRITICAL

#### `model.py` — Deteksi Anomali
- `train_isolation_forest()` — Training model dengan StandardScaler
- `evaluate_model()` — Confusion Matrix (heatmap), Precision, Recall, F1-Score
- Komentar lengkap tentang interpretasi metrik untuk keamanan data

#### `app.py` — Dashboard Streamlit
- Menggunakan `@st.cache_data` untuk performa optimal
- Sidebar filter: Departemen dan Status User
- 4 tab: Visualisasi, Deteksi Anomali, Alert Simulation, Data Explorer

---

## 🚀 Panduan Menjalankan

### Prasyarat
- Python 3.9 atau lebih baru
- pip (package manager)

### Instalasi Lokal

```bash
# 1. Clone repository
git clone https://github.com/uvuvevewewewOsas/UAS-BIG-DATA.git
cd UAS-BIG-DATA

# 2. Install dependencies
pip install -r requirements.txt

# 3. Jalankan dashboard
streamlit run app.py
```

### Deploy ke Streamlit Community Cloud

1. Push semua file ke repository GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Pilih repository `UAS-BIG-DATA`
4. Set **Main file path**: `app.py`
5. Klik **Deploy**

---

## 📊 Panduan untuk Dosen Penguji

### Cara Menguji Dashboard

1. **Buka link dashboard** (lihat bagian Live Dashboard di atas)
2. **Tab Visualisasi**: Lihat distribusi klasifikasi event, top 5 user aktif, dan timeline alert
3. **Tab Deteksi Anomali**: Ubah slider contamination rate untuk melihat pengaruhnya terhadap model
4. **Tab Real-Time Alert**: Klik "Mulai Simulasi" untuk melihat log event secara streaming
5. **Tab Data Explorer**: Jelajahi raw data dan fitur engineering

### Sidebar Filter
- Pilih **Departemen** tertentu untuk fokus analisis
- Pilih **Status User** (active/terminated) untuk investigasi spesifik

### Metrik yang Ditampilkan

| Metrik | Penjelasan |
|--------|-----------|
| **Precision** | Dari prediksi anomali, berapa % yang benar |
| **Recall** | Dari anomali sebenarnya, berapa % yang terdeteksi |
| **F1-Score** | Keseimbangan Precision & Recall |
| **Confusion Matrix** | Visualisasi TP, FP, FN, TN |

---

## 🔒 Mitigasi Risiko Keamanan

### 1. Deteksi Akses Terminated User
- Dashboard secara otomatis mengidentifikasi event dari user berstatus **terminated**
- User terminated yang masih aktif langsung mendapat alert level **CRITICAL**
- Terdapat 7 user terminated dalam dataset (U0007, U0023, U0067, U0070, U0080, U0090, U0091, U0108, U0139, U0146)

### 2. Klasifikasi Alert Multi-Level
```
🔴 CRITICAL  → Risk ≥ 80 atau label exfiltration/compromise atau user terminated
🟠 HIGH      → Risk ≥ 60 atau policy violation atau intern akses confidential
🟡 MEDIUM    → Risk ≥ 40 atau aksi sensitif (delete/permission_change)
🟢 LOW       → Aktivitas normal
```

### 3. Feature Engineering untuk Deteksi Dini
- **Failed Login Rate** tinggi → Potensi brute-force attack
- **Access to Restricted Ratio** tinggi → Unauthorized data access
- **Total Bytes Out** ekstrem → Potensi data exfiltration
- **Unique IPs** tinggi → Account sharing/compromise

### 4. Keamanan Kode
- Tidak ada API key atau token yang di-hardcode dalam kode
- Semua secret (jika diperlukan) menggunakan `os.getenv('NAMA_KEY')` atau Streamlit Secrets
- File `.gitignore` mengecualikan `.env` dan `.streamlit/secrets.toml`

---

## 📁 Dataset

| File | Deskripsi | Jumlah Record |
|------|-----------|---------------|
| `sample_stream_events.csv` | Log event keamanan sistem | 5.000 |
| `users.csv` | Profil user dengan status dan clearance | 150 |
| `assets.csv` | Daftar asset/sistem yang dimonitor | 8 |

### Kolom Utama Events

| Kolom | Deskripsi |
|-------|-----------|
| `event_id` | Identifier unik event |
| `event_time` | Waktu kejadian |
| `user_id` | ID user pelaku |
| `dept` | Departemen |
| `action` | Jenis aktivitas (login, read, query, download, dll.) |
| `status` | Berhasil/gagal |
| `risk_score` | Skor risiko (0-100) |
| `data_classification` | Level kerahasiaan (public, internal, restricted, confidential) |
| `label` | Label keamanan (normal, policy_violation, exfiltration_suspected, dll.) |

---

## 🛠️ Teknologi

- **Frontend**: Streamlit
- **Backend**: Python, Pandas, NumPy
- **Machine Learning**: scikit-learn (IsolationForest)
- **Visualisasi**: Matplotlib, Seaborn
- **Deployment**: Streamlit Community Cloud

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan UAS Perancangan Big Data.

---

<div align="center">
  <sub>Built with ❤️ using Python & Streamlit</sub>
</div>
