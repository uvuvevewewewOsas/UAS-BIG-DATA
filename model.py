"""
model.py — Modul Deteksi Anomali dengan IsolationForest
========================================================
Modul ini mengimplementasikan:
  1. Training IsolationForest untuk deteksi anomali keamanan
  2. Evaluasi model dengan Confusion Matrix, Precision, Recall, F1-Score
  3. Visualisasi hasil evaluasi (heatmap confusion matrix)

Penjelasan Metrik Penting untuk Keamanan Data:
──────────────────────────────────────────────
• Precision (Presisi):
  → Dari semua event yang diprediksi ANOMALI, berapa % yang benar-benar anomali?
  → Precision rendah = banyak false alarm → tim security kelelahan (alert fatigue)
  
• Recall (Sensitivitas):
  → Dari semua event ANOMALI sesungguhnya, berapa % yang berhasil terdeteksi?
  → Recall rendah = banyak ancaman terlewat → risiko kebocoran data tinggi
  → DALAM KONTEKS KEAMANAN: Recall LEBIH PENTING dari Precision
    karena melewatkan ancaman (False Negative) jauh lebih berbahaya
    daripada false alarm (False Positive)

• F1-Score:
  → Harmonic mean dari Precision dan Recall
  → Memberikan keseimbangan evaluasi ketika dataset tidak seimbang
  → F1 tinggi = model mampu mendeteksi anomali tanpa terlalu banyak false alarm

• Confusion Matrix:
  → True Positive (TP)  : Anomali terdeteksi dengan benar
  → True Negative (TN)  : Normal teridentifikasi dengan benar
  → False Positive (FP) : Normal salah dikategorikan sebagai anomali (false alarm)
  → False Negative (FN) : Anomali tidak terdeteksi (PALING BERBAHAYA dalam security)

IsolationForest Cocok untuk Security karena:
  → Tidak membutuhkan label (unsupervised) — cocok untuk data real-time
  → Efisien untuk dataset besar (O(n log n))
  → Mampu mendeteksi anomali tanpa asumsi distribusi data
  → Anomali akan "terisolasi" lebih cepat di dalam pohon keputusan
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend untuk Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# ============================================================================
# FITUR YANG DIGUNAKAN UNTUK MODEL
# ============================================================================

# Fitur keamanan yang dipilih berdasarkan relevansi domain:
FEATURE_COLUMNS = [
    'event_count',               # Volume aktivitas — anomali jika terlalu tinggi/rendah
    'failed_login_rate',         # Indikator brute-force atau credential stuffing
    'avg_latency',               # Bot biasanya punya latency sangat rendah/konsisten
    'access_to_restricted_ratio',# Rasio akses ke data sensitif
    'total_bytes_out',           # Volume data keluar — indikator exfiltration
    'avg_risk_score',            # Akumulasi risiko dari rule-based scoring
    'unique_assets',             # Lateral movement — mengakses banyak sistem berbeda
    'unique_ips',                # Account sharing atau compromised account
]


# ============================================================================
# 1. TRAINING MODEL
# ============================================================================

def train_isolation_forest(
    user_features: pd.DataFrame,
    contamination: float = 0.1,
    random_state: int = 42
) -> tuple:
    """
    Melatih model IsolationForest pada fitur user.

    Parameter:
      - contamination: Proporsi anomali yang diharapkan (default 10%)
        → Dalam security, biasanya 5-15% trafik adalah anomali
      - random_state: Seed untuk reproducibility

    Args:
        user_features: DataFrame dari compute_user_features()
        contamination: Persentase expected anomaly
        random_state: Random seed

    Returns:
        tuple: (model, scaler, predictions, user_features_with_pred)
          - model     : Trained IsolationForest
          - scaler    : Fitted StandardScaler (untuk transform data baru)
          - predictions: Array prediksi (-1=anomali, 1=normal)
          - df_result : DataFrame dengan kolom 'anomaly_pred'
    """
    # Pastikan semua fitur tersedia
    available_features = [f for f in FEATURE_COLUMNS if f in user_features.columns]
    X = user_features[available_features].copy()

    # Standardisasi fitur (penting untuk IsolationForest)
    # IsolationForest tetap bekerja tanpa scaling, tapi scaling membantu
    # ketika skala fitur sangat berbeda (bytes vs ratio)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Inisialisasi dan training model
    model = IsolationForest(
        n_estimators=200,          # Jumlah pohon — lebih banyak = lebih stabil
        contamination=contamination,
        max_samples='auto',        # Otomatis sesuaikan ukuran subsample
        random_state=random_state,
        n_jobs=-1,                 # Gunakan semua CPU core
    )
    model.fit(X_scaled)

    # Prediksi: -1 = anomali, 1 = normal
    predictions = model.predict(X_scaled)

    # Anomaly scores (semakin negatif = semakin anomali)
    scores = model.decision_function(X_scaled)

    # Tambahkan hasil ke DataFrame
    df_result = user_features.copy()
    df_result['anomaly_pred'] = predictions
    df_result['anomaly_score'] = scores
    df_result['classification'] = df_result['anomaly_pred'].map(
        {1: 'Normal', -1: 'Anomaly'}
    )

    return model, scaler, predictions, df_result


# ============================================================================
# 2. EVALUASI MODEL
# ============================================================================

def create_ground_truth(df_events: pd.DataFrame, user_features: pd.DataFrame) -> pd.Series:
    """
    Membuat ground truth label per user berdasarkan label events.

    Logika:
      → Jika user memiliki SATU SAJA event berlabel anomali
        (policy_violation, exfiltration_suspected, compromised_account, privilege_abuse),
        maka user tersebut dianggap anomali.
      → Ini adalah pendekatan konservatif yang sesuai prinsip keamanan:
        "satu pelanggaran sudah cukup untuk di-flag"

    Args:
        df_events: DataFrame events asli (dengan kolom 'label')
        user_features: DataFrame fitur per user

    Returns:
        Series ground_truth: -1 (anomali) atau 1 (normal) per user
    """
    # Definisikan label anomali
    anomaly_labels = ['policy_violation', 'exfiltration_suspected',
                      'compromised_account', 'privilege_abuse']

    # Cek apakah tiap user punya minimal 1 event anomali
    anomaly_users = df_events[
        df_events['label'].isin(anomaly_labels)
    ]['user_id'].unique()

    # Buat mapping: -1 jika anomali, 1 jika normal
    ground_truth = user_features['user_id'].apply(
        lambda uid: -1 if uid in anomaly_users else 1
    )

    return ground_truth


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list = None
) -> dict:
    """
    Evaluasi model anomaly detection dan generate semua metrik.

    Args:
        y_true: Ground truth labels (-1 atau 1)
        y_pred: Predicted labels (-1 atau 1)
        labels: Optional list of label values

    Returns:
        dict berisi:
          - confusion_matrix: np.ndarray 2x2
          - precision: float
          - recall: float
          - f1_score: float
          - classification_report: string
          - fig_confusion: matplotlib Figure (heatmap)

    Catatan Keamanan:
    ─────────────────
    • Dalam konteks ini, label POSITIF = anomali (-1)
    • False Negative (FN) = anomali yang TIDAK terdeteksi
      → Ini risiko terbesar: serangan lolos tanpa peringatan
    • False Positive (FP) = event normal yang di-flag anomali
      → Menyebabkan alert fatigue, tapi masih bisa ditoleransi
    """
    if labels is None:
        labels = [-1, 1]

    # Konversi ke numpy array
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Hitung confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Hitung metrik (pos_label=-1 karena anomali = -1)
    precision = precision_score(y_true, y_pred, pos_label=-1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=-1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=-1, zero_division=0)

    # Classification report string
    target_names = ['Anomaly (-1)', 'Normal (1)']
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0
    )

    # Buat heatmap confusion matrix
    fig_cm = plot_confusion_matrix(cm, target_names)

    return {
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report,
        'fig_confusion': fig_cm,
    }


def plot_confusion_matrix(cm: np.ndarray, labels: list) -> plt.Figure:
    """
    Membuat heatmap Confusion Matrix dengan Seaborn.

    Interpretasi warna:
      → Diagonal (kiri atas & kanan bawah) = prediksi BENAR
      → Off-diagonal = prediksi SALAH
      → Semakin gelap warna = semakin banyak count

    Args:
        cm: Confusion matrix array
        labels: List nama label ['Anomaly', 'Normal']

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='YlOrRd',              # Skema warna: Kuning → Oranye → Merah
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        linewidths=2,
        linecolor='white',
        cbar_kws={'label': 'Jumlah Prediksi'},
        annot_kws={'size': 18, 'weight': 'bold'},
        ax=ax,
    )

    ax.set_xlabel('Prediksi Model', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Label Sebenarnya (Ground Truth)', fontsize=13,
                  fontweight='bold', labelpad=10)
    ax.set_title('Confusion Matrix — Deteksi Anomali Keamanan',
                 fontsize=15, fontweight='bold', pad=15)

    # Tambahkan anotasi penjelasan
    fig.text(
        0.5, -0.02,
        'TP = Anomali terdeteksi benar  |  FP = Normal salah di-flag  |  '
        'FN = Anomali terlewat (BAHAYA)  |  TN = Normal teridentifikasi benar',
        ha='center', fontsize=9, style='italic', color='#555555'
    )

    plt.tight_layout()
    return fig


# ============================================================================
# 3. UTILITY: PREDIKSI EVENT BARU
# ============================================================================

def predict_new_user(model, scaler, user_feature_row: pd.DataFrame) -> str:
    """
    Memprediksi apakah user baru termasuk anomali atau normal.

    Args:
        model: Trained IsolationForest
        scaler: Fitted StandardScaler
        user_feature_row: DataFrame 1 baris berisi fitur user

    Returns:
        'Anomaly' atau 'Normal'
    """
    available_features = [f for f in FEATURE_COLUMNS if f in user_feature_row.columns]
    X = user_feature_row[available_features].values
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    return 'Anomaly' if pred[0] == -1 else 'Normal'
