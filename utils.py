"""
utils.py — Modul Pembersihan Data & Feature Engineering
========================================================
Modul ini berisi fungsi-fungsi utilitas untuk:
  1. Pembersihan dan validasi data mentah dari CSV
  2. Feature Engineering per user untuk model anomali
  3. Klasifikasi alert keamanan berdasarkan event
  4. Merge data events dengan informasi user

Digunakan oleh: app.py (dashboard) dan model.py (training)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


# ============================================================================
# 1. FUNGSI PEMBERSIHAN DATA (Data Cleaning)
# ============================================================================

def load_data():
    """
    Memuat dataset dari file CSV.
    Mendukung path relatif (untuk Streamlit Cloud) dan absolut.

    Returns:
        tuple: (df_events, df_users, df_assets) — tiga DataFrame utama
    """
    # Tentukan base directory (lokasi file ini)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    df_events = pd.read_csv(os.path.join(base_dir, 'sample_stream_events.csv'))
    df_users = pd.read_csv(os.path.join(base_dir, 'users.csv'))
    df_assets = pd.read_csv(os.path.join(base_dir, 'assets.csv'))

    return df_events, df_users, df_assets


def clean_events(df_events: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan dan memvalidasi DataFrame events.

    Langkah-langkah:
      - Konversi event_time ke datetime
      - Hapus duplikat berdasarkan event_id
      - Isi missing values pada kolom numerik dengan 0
      - Pastikan tipe data konsisten
      - Tambahkan kolom turunan waktu (hour, date)

    Args:
        df_events: DataFrame mentah dari sample_stream_events.csv

    Returns:
        DataFrame yang sudah dibersihkan dan siap diproses
    """
    df = df_events.copy()

    # Konversi event_time ke datetime
    df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')

    # Hapus duplikat event_id (jaga entry pertama)
    df = df.drop_duplicates(subset=['event_id'], keep='first')

    # Isi NaN pada kolom numerik dengan 0
    numeric_cols = ['bytes_out', 'records_accessed', 'latency_ms', 'risk_score']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Isi NaN pada kolom string dengan 'unknown'
    string_cols = ['user_id', 'dept', 'role', 'device_type', 'source_ip',
                   'asset_id', 'asset_type', 'data_classification', 'action',
                   'status', 'label']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')

    # Tambahkan kolom turunan waktu untuk analisis temporal
    df['hour'] = df['event_time'].dt.hour
    df['date'] = df['event_time'].dt.date

    return df


def clean_users(df_users: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan DataFrame users.

    Args:
        df_users: DataFrame mentah dari users.csv

    Returns:
        DataFrame users yang sudah bersih
    """
    df = df_users.copy()
    df = df.drop_duplicates(subset=['user_id'], keep='first')

    string_cols = ['user_id', 'dept', 'role', 'clearance', 'status']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].fillna('unknown').str.strip()

    return df


def merge_events_users(df_events: pd.DataFrame, df_users: pd.DataFrame) -> pd.DataFrame:
    """
    Menggabungkan events dengan informasi user (termasuk status user).
    Kolom 'status' dari users di-rename menjadi 'user_status' untuk
    menghindari konflik dengan kolom 'status' (success/failed) pada events.

    Args:
        df_events: DataFrame events yang sudah bersih
        df_users: DataFrame users yang sudah bersih

    Returns:
        DataFrame gabungan dengan kolom user_status dan clearance
    """
    # Pilih kolom yang relevan dari users
    user_cols = ['user_id', 'clearance', 'status', 'location']
    available_cols = [c for c in user_cols if c in df_users.columns]
    users_subset = df_users[available_cols].copy()

    # Rename status agar tidak konflik
    if 'status' in users_subset.columns:
        users_subset = users_subset.rename(columns={'status': 'user_status'})

    # Left join — semua events tetap ada
    df_merged = df_events.merge(users_subset, on='user_id', how='left')

    # Isi NaN hasil join
    if 'user_status' in df_merged.columns:
        df_merged['user_status'] = df_merged['user_status'].fillna('unknown')
    if 'clearance' in df_merged.columns:
        df_merged['clearance'] = df_merged['clearance'].fillna('unknown')

    return df_merged


# ============================================================================
# 2. FEATURE ENGINEERING — Ekstraksi Fitur Keamanan per User
# ============================================================================

def compute_user_features(df_events: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung fitur-fitur keamanan agregat per user_id.
    Fitur ini digunakan sebagai input untuk model IsolationForest.

    Fitur yang dihitung:
      1. event_count        : Total aktivitas user
      2. failed_login_rate  : Rasio login gagal / total login
                              → Indikator brute-force attack
      3. avg_latency        : Rata-rata waktu respons (ms)
                              → Anomali kinerja / bot traffic
      4. access_to_restricted_ratio : Rasio akses ke data restricted+confidential
                              → Indikator unauthorized data access
      5. total_bytes_out    : Total data yang dikirim keluar
                              → Indikator data exfiltration
      6. avg_risk_score     : Rata-rata risk score
                              → Indikator akumulasi risiko
      7. unique_assets      : Jumlah asset unik yang diakses
                              → Pola lateral movement
      8. unique_ips         : Jumlah IP unik yang digunakan
                              → Indikator account sharing / compromise

    Args:
        df_events: DataFrame events yang sudah bersih

    Returns:
        DataFrame user_features dengan satu baris per user_id
    """
    df = df_events.copy()

    # --- 1. Total event count per user ---
    user_features = df.groupby('user_id').size().reset_index(name='event_count')

    # --- 2. Failed Login Rate ---
    # Hitung login gagal per user
    login_events = df[df['action'] == 'login']
    total_logins = login_events.groupby('user_id').size()
    failed_logins = login_events[login_events['status'] == 'failed'].groupby('user_id').size()

    # Hitung rasio: failed / total (handle divide-by-zero)
    failed_login_rate = (failed_logins / total_logins).fillna(0)
    user_features['failed_login_rate'] = user_features['user_id'].map(failed_login_rate).fillna(0)

    # --- 3. Average Latency ---
    avg_latency = df.groupby('user_id')['latency_ms'].mean()
    user_features['avg_latency'] = user_features['user_id'].map(avg_latency).fillna(0)

    # --- 4. Access to Restricted/Confidential Ratio ---
    restricted_mask = df['data_classification'].isin(['restricted', 'confidential'])
    restricted_count = df[restricted_mask].groupby('user_id').size()
    total_count = df.groupby('user_id').size()
    restricted_ratio = (restricted_count / total_count).fillna(0)
    user_features['access_to_restricted_ratio'] = user_features['user_id'].map(restricted_ratio).fillna(0)

    # --- 5. Total Bytes Out ---
    total_bytes = df.groupby('user_id')['bytes_out'].sum()
    user_features['total_bytes_out'] = user_features['user_id'].map(total_bytes).fillna(0)

    # --- 6. Average Risk Score ---
    avg_risk = df.groupby('user_id')['risk_score'].mean()
    user_features['avg_risk_score'] = user_features['user_id'].map(avg_risk).fillna(0)

    # --- 7. Unique Assets Accessed ---
    unique_assets = df.groupby('user_id')['asset_id'].nunique()
    user_features['unique_assets'] = user_features['user_id'].map(unique_assets).fillna(0)

    # --- 8. Unique Source IPs ---
    unique_ips = df.groupby('user_id')['source_ip'].nunique()
    user_features['unique_ips'] = user_features['user_id'].map(unique_ips).fillna(0)

    return user_features


# ============================================================================
# 3. SECURITY ALERT CLASSIFICATION
# ============================================================================

def security_alert(event: dict) -> dict:
    """
    Mengklasifikasikan satu event keamanan menjadi level alert:
      - CRITICAL : risk_score >= 80  ATAU label bersifat eksfiltrasi/compromise
      - HIGH     : risk_score >= 60  ATAU label policy_violation
      - MEDIUM   : risk_score >= 40  ATAU aksi sensitif (delete, permission_change)
      - LOW      : Kondisi lainnya

    Logika Bisnis:
      → Terminated user yang masih aktif = langsung CRITICAL
      → Akses ke data confidential oleh intern = minimal HIGH
      → Anomali volume data (bytes_out > 200000) menaikkan level

    Args:
        event: Dictionary berisi data satu event

    Returns:
        Dictionary event + field 'alert_level' dan 'alert_reason'
    """
    result = dict(event)
    risk = int(event.get('risk_score', 0))
    label = str(event.get('label', 'normal'))
    action = str(event.get('action', ''))
    user_status = str(event.get('user_status', 'active'))
    data_class = str(event.get('data_classification', 'public'))
    role = str(event.get('role', ''))
    bytes_out = int(event.get('bytes_out', 0))
    status = str(event.get('status', 'success'))

    alert_level = 'LOW'
    reasons = []

    # --- CRITICAL conditions ---
    if label in ['exfiltration_suspected', 'compromised_account']:
        alert_level = 'CRITICAL'
        reasons.append(f'Label terdeteksi: {label}')

    if user_status == 'terminated':
        alert_level = 'CRITICAL'
        reasons.append('User berstatus TERMINATED masih melakukan aktivitas')

    if risk >= 80:
        alert_level = 'CRITICAL'
        reasons.append(f'Risk score sangat tinggi: {risk}')

    # --- HIGH conditions ---
    if alert_level not in ['CRITICAL']:
        if label == 'policy_violation':
            alert_level = 'HIGH'
            reasons.append(f'Policy violation terdeteksi')

        if label == 'privilege_abuse':
            alert_level = 'HIGH'
            reasons.append(f'Privilege abuse terdeteksi')

        if risk >= 60:
            alert_level = 'HIGH'
            reasons.append(f'Risk score tinggi: {risk}')

        if role == 'intern' and data_class in ['confidential', 'restricted']:
            alert_level = 'HIGH'
            reasons.append(f'Intern mengakses data {data_class}')

    # --- MEDIUM conditions ---
    if alert_level not in ['CRITICAL', 'HIGH']:
        if risk >= 40:
            alert_level = 'MEDIUM'
            reasons.append(f'Risk score sedang: {risk}')

        if action in ['delete', 'permission_change']:
            alert_level = 'MEDIUM'
            reasons.append(f'Aksi sensitif: {action}')

        if bytes_out > 200000:
            alert_level = 'MEDIUM'
            reasons.append(f'Volume data besar: {bytes_out:,} bytes')

        if status == 'failed' and action == 'login':
            alert_level = 'MEDIUM'
            reasons.append('Login gagal terdeteksi')

    # --- LOW (default) ---
    if not reasons:
        reasons.append('Aktivitas normal')

    result['alert_level'] = alert_level
    result['alert_reason'] = ' | '.join(reasons)

    return result


def classify_all_alerts(df_events: pd.DataFrame) -> pd.DataFrame:
    """
    Menerapkan fungsi security_alert ke seluruh DataFrame.

    Args:
        df_events: DataFrame events (sudah di-merge dengan user info)

    Returns:
        DataFrame dengan kolom tambahan alert_level dan alert_reason
    """
    alerts = df_events.apply(
        lambda row: security_alert(row.to_dict()), axis=1, result_type='expand'
    )

    # Hanya ambil kolom baru
    df_result = df_events.copy()
    df_result['alert_level'] = alerts['alert_level']
    df_result['alert_reason'] = alerts['alert_reason']

    return df_result


# ============================================================================
# 4. HELPER FUNCTIONS
# ============================================================================

def get_alert_color(level: str) -> str:
    """Mengembalikan warna hex berdasarkan alert level."""
    colors = {
        'CRITICAL': '#FF1744',  # Merah terang
        'HIGH':     '#FF9100',  # Oranye
        'MEDIUM':   '#FFD600',  # Kuning
        'LOW':      '#00E676',  # Hijau
    }
    return colors.get(level, '#90A4AE')


def get_alert_emoji(level: str) -> str:
    """Mengembalikan emoji berdasarkan alert level."""
    emojis = {
        'CRITICAL': '🔴',
        'HIGH':     '🟠',
        'MEDIUM':   '🟡',
        'LOW':      '🟢',
    }
    return emojis.get(level, '⚪')


def format_alert_log(event: dict) -> str:
    """
    Format satu event menjadi string log yang readable.

    Args:
        event: Dictionary event dengan alert info

    Returns:
        String formatted log entry
    """
    emoji = get_alert_emoji(event.get('alert_level', 'LOW'))
    time_str = str(event.get('event_time', 'N/A'))[:19]
    user = event.get('user_id', 'N/A')
    action = event.get('action', 'N/A')
    asset = event.get('asset_id', 'N/A')
    level = event.get('alert_level', 'LOW')
    reason = event.get('alert_reason', '')

    return (
        f"{emoji} [{level}] {time_str} — "
        f"User: {user} | Action: {action} | Asset: {asset}\n"
        f"   └─ {reason}"
    )
