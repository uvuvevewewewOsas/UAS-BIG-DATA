"""
app.py — Dashboard Utama Streamlit: Big Data Security Analytics
================================================================
UAS Perancangan Big Data

Dashboard ini menyediakan:
  1. Overview metrik keamanan utama
  2. Deteksi anomali menggunakan IsolationForest
  3. Visualisasi distribusi klasifikasi, top users, dan alert timeline
  4. Simulasi real-time alert log
  5. Filter interaktif berdasarkan departemen dan status user

Deploy: Streamlit Community Cloud
Author: Mahasiswa UAS Perancangan Big Data
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from datetime import datetime

# Import modul lokal
from utils import (
    load_data, clean_events, clean_users, merge_events_users,
    compute_user_features, classify_all_alerts,
    security_alert, get_alert_color, get_alert_emoji, format_alert_log
)
from model import (
    train_isolation_forest, create_ground_truth, evaluate_model,
    FEATURE_COLUMNS
)

# ============================================================================
# KONFIGURASI HALAMAN
# ============================================================================

st.set_page_config(
    page_title="🛡️ Big Data Security Analytics Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS — Desain Premium
# ============================================================================

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .main-header h1 {
        color: #FFFFFF;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #B0BEC5;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.3rem 0;
        letter-spacing: -1px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #78909C;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* Alert badges */
    .alert-critical { color: #FF1744; }
    .alert-high { color: #FF9100; }
    .alert-medium { color: #FFD600; }
    .alert-low { color: #00E676; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1a1a2e, transparent);
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #7C4DFF;
        margin: 1.5rem 0 1rem 0;
    }
    .section-header h3 {
        color: #E0E0E0;
        margin: 0;
        font-size: 1.15rem;
        font-weight: 700;
    }

    /* Log container */
    .log-container {
        background: #0a0a1a;
        border: 1px solid #1a1a3e;
        border-radius: 10px;
        padding: 1rem;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.8rem;
        max-height: 400px;
        overflow-y: auto;
        line-height: 1.6;
    }

    /* Table styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }
    [data-testid="stSidebar"] .css-1d391kg {
        padding: 2rem 1rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADING WITH CACHING
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_and_process_data():
    """
    Memuat, membersihkan, dan memproses semua data.
    Menggunakan @st.cache_data untuk performa optimal.
    """
    # Load raw data
    df_events, df_users, df_assets = load_data()

    # Clean data
    df_events = clean_events(df_events)
    df_users = clean_users(df_users)

    # Merge events + users
    df_merged = merge_events_users(df_events, df_users)

    # Classify alerts
    df_alerts = classify_all_alerts(df_merged)

    # Compute user features
    user_features = compute_user_features(df_events)

    return df_events, df_users, df_assets, df_merged, df_alerts, user_features


@st.cache_data(ttl=3600, show_spinner=False)
def run_anomaly_detection(user_features_json: str, events_json: str, contamination: float):
    """
    Menjalankan IsolationForest dan evaluasi model.
    Input di-serialize ke JSON string agar hashable oleh st.cache_data.
    """
    user_features = pd.read_json(user_features_json)
    df_events = pd.read_json(events_json)

    # Train model
    model, scaler, predictions, df_result = train_isolation_forest(
        user_features, contamination=contamination
    )

    # Create ground truth
    ground_truth = create_ground_truth(df_events, user_features)

    # Evaluate
    eval_results = evaluate_model(ground_truth.values, predictions)

    return df_result, eval_results


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Fungsi utama dashboard."""

    # --- Load Data ---
    with st.spinner('🔄 Memuat dan memproses data...'):
        df_events, df_users, df_assets, df_merged, df_alerts, user_features = load_and_process_data()

    # ========================================================================
    # SIDEBAR — Filters & Controls
    # ========================================================================
    with st.sidebar:
        st.markdown("## 🛡️ Control Panel")
        st.markdown("---")

        # Filter: Departemen
        st.markdown("### 📂 Filter Departemen")
        all_depts = sorted(df_alerts['dept'].unique())
        selected_depts = st.multiselect(
            "Pilih Departemen:",
            options=all_depts,
            default=all_depts,
            key="dept_filter"
        )

        st.markdown("---")

        # Filter: Status User
        st.markdown("### 👤 Filter Status User")
        if 'user_status' in df_alerts.columns:
            all_statuses = sorted(df_alerts['user_status'].unique())
        else:
            all_statuses = ['active', 'terminated']
        selected_statuses = st.multiselect(
            "Pilih Status User:",
            options=all_statuses,
            default=all_statuses,
            key="status_filter"
        )

        st.markdown("---")

        # Model parameter
        st.markdown("### ⚙️ Parameter Model")
        contamination = st.slider(
            "Contamination Rate",
            min_value=0.01,
            max_value=0.30,
            value=0.10,
            step=0.01,
            help="Proporsi anomali yang diharapkan dalam data (default: 10%)"
        )

        st.markdown("---")
        st.markdown(
            """
            <div style="text-align:center; color:#78909C; font-size:0.75rem;">
                <p>📊 UAS Perancangan Big Data</p>
                <p>Streamlit Community Cloud</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- Apply Filters ---
    df_filtered = df_alerts.copy()
    if selected_depts:
        df_filtered = df_filtered[df_filtered['dept'].isin(selected_depts)]
    if selected_statuses and 'user_status' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['user_status'].isin(selected_statuses)]

    # ========================================================================
    # HEADER
    # ========================================================================
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Big Data Security Analytics Dashboard</h1>
        <p>Sistem Deteksi Anomali & Monitoring Keamanan Data — UAS Perancangan Big Data</p>
    </div>
    """, unsafe_allow_html=True)

    # ========================================================================
    # METRIC CARDS — Overview
    # ========================================================================
    total_events = len(df_filtered)
    total_users = df_filtered['user_id'].nunique()
    critical_alerts = len(df_filtered[df_filtered['alert_level'] == 'CRITICAL'])
    high_alerts = len(df_filtered[df_filtered['alert_level'] == 'HIGH'])
    terminated_active = len(df_filtered[
        (df_filtered.get('user_status', pd.Series()) == 'terminated')
    ]) if 'user_status' in df_filtered.columns else 0

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Events</div>
            <div class="metric-value" style="color: #64B5F6;">{total_events:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Unique Users</div>
            <div class="metric-value" style="color: #81C784;">{total_users}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🔴 Critical Alerts</div>
            <div class="metric-value alert-critical">{critical_alerts}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🟠 High Alerts</div>
            <div class="metric-value alert-high">{high_alerts}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚠️ Terminated Active</div>
            <div class="metric-value" style="color: #FF5252;">{terminated_active}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ========================================================================
    # TABS — Visualisasi, Model, Alerting
    # ========================================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Visualisasi & Analisis",
        "🤖 Deteksi Anomali (IsolationForest)",
        "🚨 Real-Time Alert Simulation",
        "📋 Data Explorer"
    ])

    # ====================================================================
    # TAB 1: VISUALISASI
    # ====================================================================
    with tab1:
        st.markdown("""
        <div class="section-header">
            <h3>📊 Analisis Visual Keamanan Data</h3>
        </div>
        """, unsafe_allow_html=True)

        # --- Row 1: Classification Distribution + Alert Distribution ---
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            st.markdown("#### 🏷️ Distribusi Klasifikasi Event")
            fig1, ax1 = plt.subplots(figsize=(8, 5))

            label_counts = df_filtered['label'].value_counts()
            colors_map = {
                'normal': '#00E676',
                'policy_violation': '#FF9100',
                'exfiltration_suspected': '#FF1744',
                'compromised_account': '#D500F9',
                'privilege_abuse': '#FFD600',
            }
            bar_colors = [colors_map.get(l, '#90A4AE') for l in label_counts.index]

            bars = ax1.barh(label_counts.index, label_counts.values, color=bar_colors,
                           edgecolor='white', linewidth=0.5, height=0.6)

            # Tambahkan label angka di bar
            for bar, val in zip(bars, label_counts.values):
                ax1.text(bar.get_width() + max(label_counts.values)*0.02,
                        bar.get_y() + bar.get_height()/2,
                        f'{val:,}', va='center', fontweight='bold',
                        fontsize=11, color='white')

            ax1.set_facecolor('#0a0a1a')
            fig1.set_facecolor('#0a0a1a')
            ax1.tick_params(colors='white')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_color('#333')
            ax1.spines['left'].set_color('#333')
            for label in ax1.get_yticklabels():
                label.set_color('white')
                label.set_fontsize(10)
            for label in ax1.get_xticklabels():
                label.set_color('white')
            ax1.set_xlabel('Jumlah Event', color='white', fontsize=11)
            ax1.set_title('Distribusi Label Klasifikasi', color='white',
                         fontsize=14, fontweight='bold', pad=12)
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

        with viz_col2:
            st.markdown("#### 🚦 Distribusi Alert Level")
            fig2, ax2 = plt.subplots(figsize=(8, 5))

            alert_counts = df_filtered['alert_level'].value_counts()
            alert_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
            alert_counts = alert_counts.reindex([a for a in alert_order if a in alert_counts.index])

            alert_colors = {
                'CRITICAL': '#FF1744',
                'HIGH': '#FF9100',
                'MEDIUM': '#FFD600',
                'LOW': '#00E676',
            }
            pie_colors = [alert_colors.get(l, '#90A4AE') for l in alert_counts.index]

            wedges, texts, autotexts = ax2.pie(
                alert_counts.values,
                labels=alert_counts.index,
                colors=pie_colors,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'color': 'white', 'fontweight': 'bold', 'fontsize': 11},
                wedgeprops={'edgecolor': '#0a0a1a', 'linewidth': 2},
                pctdistance=0.75,
            )
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_color('white')

            # Donut effect
            centre_circle = plt.Circle((0, 0), 0.55, fc='#0a0a1a')
            ax2.add_artist(centre_circle)

            ax2.set_facecolor('#0a0a1a')
            fig2.set_facecolor('#0a0a1a')
            ax2.set_title('Distribusi Alert Level', color='white',
                         fontsize=14, fontweight='bold', pad=12)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        # --- Row 2: Top 5 Users + Alert Frequency Timeline ---
        st.markdown("<br>", unsafe_allow_html=True)
        viz_col3, viz_col4 = st.columns(2)

        with viz_col3:
            st.markdown("#### 👥 Top 5 User Paling Aktif")
            fig3, ax3 = plt.subplots(figsize=(8, 5))

            top_users = df_filtered['user_id'].value_counts().head(5)
            gradient_colors = ['#7C4DFF', '#651FFF', '#536DFE', '#448AFF', '#42A5F5']

            bars3 = ax3.bar(top_users.index, top_users.values,
                          color=gradient_colors, edgecolor='white',
                          linewidth=0.5, width=0.6)

            for bar, val in zip(bars3, top_users.values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(val), ha='center', fontweight='bold',
                        fontsize=12, color='white')

            ax3.set_facecolor('#0a0a1a')
            fig3.set_facecolor('#0a0a1a')
            ax3.tick_params(colors='white')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['bottom'].set_color('#333')
            ax3.spines['left'].set_color('#333')
            for label in ax3.get_xticklabels() + ax3.get_yticklabels():
                label.set_color('white')
                label.set_fontsize(10)
            ax3.set_xlabel('User ID', color='white', fontsize=11)
            ax3.set_ylabel('Jumlah Event', color='white', fontsize=11)
            ax3.set_title('5 User Paling Aktif', color='white',
                         fontsize=14, fontweight='bold', pad=12)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

        with viz_col4:
            st.markdown("#### 📈 Frekuensi Alert per Waktu (Timeline)")
            fig4, ax4 = plt.subplots(figsize=(8, 5))

            # Buat timeline per jam
            if 'hour' in df_filtered.columns:
                non_low = df_filtered[df_filtered['alert_level'] != 'LOW']
                if len(non_low) > 0:
                    alert_timeline = non_low.groupby(['hour', 'alert_level']).size().unstack(fill_value=0)

                    timeline_colors = {
                        'CRITICAL': '#FF1744',
                        'HIGH': '#FF9100',
                        'MEDIUM': '#FFD600',
                    }

                    for level in ['MEDIUM', 'HIGH', 'CRITICAL']:
                        if level in alert_timeline.columns:
                            ax4.plot(alert_timeline.index, alert_timeline[level],
                                   marker='o', linewidth=2.5, markersize=6,
                                   color=timeline_colors.get(level, '#90A4AE'),
                                   label=level, alpha=0.9)
                            ax4.fill_between(alert_timeline.index, alert_timeline[level],
                                           alpha=0.1, color=timeline_colors.get(level))

                    ax4.legend(facecolor='#1a1a2e', edgecolor='#333',
                             labelcolor='white', fontsize=10)
                else:
                    ax4.text(0.5, 0.5, 'Tidak ada alert non-LOW',
                           transform=ax4.transAxes, ha='center', va='center',
                           color='#78909C', fontsize=12)

            ax4.set_facecolor('#0a0a1a')
            fig4.set_facecolor('#0a0a1a')
            ax4.tick_params(colors='white')
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
            ax4.spines['bottom'].set_color('#333')
            ax4.spines['left'].set_color('#333')
            for label in ax4.get_xticklabels() + ax4.get_yticklabels():
                label.set_color('white')
            ax4.set_xlabel('Jam (Hour)', color='white', fontsize=11)
            ax4.set_ylabel('Jumlah Alert', color='white', fontsize=11)
            ax4.set_title('Frekuensi Alert per Jam', color='white',
                         fontsize=14, fontweight='bold', pad=12)
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)

        # --- Row 3: Department Breakdown ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🏢 Alert per Departemen")

        fig5, ax5 = plt.subplots(figsize=(14, 5))
        dept_alert = df_filtered.groupby(['dept', 'alert_level']).size().unstack(fill_value=0)
        dept_alert_ordered = dept_alert.reindex(
            columns=[c for c in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'] if c in dept_alert.columns]
        )

        dept_colors = ['#00E676', '#FFD600', '#FF9100', '#FF1744']
        dept_alert_ordered.plot(kind='bar', stacked=True, ax=ax5,
                                color=dept_colors[:len(dept_alert_ordered.columns)],
                                edgecolor='#0a0a1a', linewidth=0.5)

        ax5.set_facecolor('#0a0a1a')
        fig5.set_facecolor('#0a0a1a')
        ax5.tick_params(colors='white')
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.spines['bottom'].set_color('#333')
        ax5.spines['left'].set_color('#333')
        for label in ax5.get_xticklabels() + ax5.get_yticklabels():
            label.set_color('white')
        ax5.set_xlabel('Departemen', color='white', fontsize=11)
        ax5.set_ylabel('Jumlah Event', color='white', fontsize=11)
        ax5.set_title('Distribusi Alert Level per Departemen', color='white',
                     fontsize=14, fontweight='bold', pad=12)
        ax5.legend(facecolor='#1a1a2e', edgecolor='#333',
                  labelcolor='white', fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

    # ====================================================================
    # TAB 2: DETEKSI ANOMALI
    # ====================================================================
    with tab2:
        st.markdown("""
        <div class="section-header">
            <h3>🤖 Deteksi Anomali dengan IsolationForest</h3>
        </div>
        """, unsafe_allow_html=True)

        st.info(
            "**IsolationForest** bekerja dengan prinsip isolasi: data anomali lebih mudah "
            "diisolasi karena memiliki pola yang berbeda dari mayoritas. Semakin sedikit "
            "split yang diperlukan, semakin anomali data tersebut."
        )

        # Run model
        with st.spinner('🔄 Melatih model IsolationForest...'):
            df_result, eval_results = run_anomaly_detection(
                user_features.to_json(),
                df_events.to_json(),
                contamination
            )

        # --- Model Results Overview ---
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        n_anomaly = len(df_result[df_result['classification'] == 'Anomaly'])
        n_normal = len(df_result[df_result['classification'] == 'Normal'])

        with res_col1:
            st.metric("Total Users Analyzed", len(df_result))
        with res_col2:
            st.metric("🔴 Anomali Terdeteksi", n_anomaly)
        with res_col3:
            st.metric("🟢 Normal", n_normal)
        with res_col4:
            st.metric("Contamination Rate", f"{contamination:.0%}")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Histogram Distribution + Confusion Matrix ---
        model_col1, model_col2 = st.columns(2)

        with model_col1:
            st.markdown("#### 📊 Histogram Distribusi Klasifikasi")
            fig_hist, ax_hist = plt.subplots(figsize=(8, 5))

            class_counts = df_result['classification'].value_counts()
            hist_colors = {'Normal': '#00E676', 'Anomaly': '#FF1744'}
            bar_colors_h = [hist_colors.get(c, '#90A4AE') for c in class_counts.index]

            bars_h = ax_hist.bar(class_counts.index, class_counts.values,
                               color=bar_colors_h, edgecolor='white',
                               linewidth=0.5, width=0.5)

            for bar, val in zip(bars_h, class_counts.values):
                ax_hist.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           str(val), ha='center', fontweight='bold',
                           fontsize=14, color='white')

            ax_hist.set_facecolor('#0a0a1a')
            fig_hist.set_facecolor('#0a0a1a')
            ax_hist.tick_params(colors='white')
            ax_hist.spines['top'].set_visible(False)
            ax_hist.spines['right'].set_visible(False)
            ax_hist.spines['bottom'].set_color('#333')
            ax_hist.spines['left'].set_color('#333')
            for label in ax_hist.get_xticklabels() + ax_hist.get_yticklabels():
                label.set_color('white')
                label.set_fontsize(11)
            ax_hist.set_ylabel('Jumlah User', color='white', fontsize=11)
            ax_hist.set_title('Distribusi Klasifikasi User', color='white',
                            fontsize=14, fontweight='bold', pad=12)
            plt.tight_layout()
            st.pyplot(fig_hist)
            plt.close(fig_hist)

        with model_col2:
            st.markdown("#### 🎯 Confusion Matrix")
            st.pyplot(eval_results['fig_confusion'])
            plt.close(eval_results['fig_confusion'])

        # --- Metrik Evaluasi ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📏 Metrik Evaluasi Model")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            precision_pct = eval_results['precision'] * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Precision</div>
                <div class="metric-value" style="color: #64B5F6;">{precision_pct:.1f}%</div>
                <p style="color:#78909C; font-size:0.75rem; margin:0.3rem 0 0 0;">
                    Ketepatan prediksi anomali
                </p>
            </div>
            """, unsafe_allow_html=True)

        with metric_col2:
            recall_pct = eval_results['recall'] * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value" style="color: #81C784;">{recall_pct:.1f}%</div>
                <p style="color:#78909C; font-size:0.75rem; margin:0.3rem 0 0 0;">
                    Cakupan deteksi anomali
                </p>
            </div>
            """, unsafe_allow_html=True)

        with metric_col3:
            f1_pct = eval_results['f1_score'] * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value" style="color: #FFD54F;">{f1_pct:.1f}%</div>
                <p style="color:#78909C; font-size:0.75rem; margin:0.3rem 0 0 0;">
                    Harmonic mean Precision & Recall
                </p>
            </div>
            """, unsafe_allow_html=True)

        # --- Classification Report ---
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📄 Lihat Classification Report Lengkap"):
            st.code(eval_results['classification_report'], language='text')

        # --- Anomaly Users Table ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔍 Daftar User Anomali")
        anomaly_users = df_result[df_result['classification'] == 'Anomaly'].sort_values(
            'anomaly_score', ascending=True
        )
        if len(anomaly_users) > 0:
            display_cols = ['user_id', 'event_count', 'failed_login_rate',
                          'avg_latency', 'access_to_restricted_ratio',
                          'total_bytes_out', 'avg_risk_score', 'anomaly_score',
                          'classification']
            available_display = [c for c in display_cols if c in anomaly_users.columns]
            st.dataframe(
                anomaly_users[available_display].style.format({
                    'failed_login_rate': '{:.3f}',
                    'avg_latency': '{:.1f}',
                    'access_to_restricted_ratio': '{:.3f}',
                    'total_bytes_out': '{:,.0f}',
                    'avg_risk_score': '{:.1f}',
                    'anomaly_score': '{:.4f}',
                }).applymap(
                    lambda v: 'color: #FF1744; font-weight: bold'
                    if v == 'Anomaly' else '', subset=['classification']
                ),
                use_container_width=True,
                height=400,
            )
        else:
            st.success("✅ Tidak ada user anomali terdeteksi dengan contamination rate ini.")

    # ====================================================================
    # TAB 3: REAL-TIME ALERT SIMULATION
    # ====================================================================
    with tab3:
        st.markdown("""
        <div class="section-header">
            <h3>🚨 Simulasi Real-Time Alert Log</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "Simulasi ini menampilkan event log secara real-time, "
            "mirip dengan SIEM (Security Information and Event Management) dashboard."
        )

        # Controls
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            alert_filter = st.selectbox(
                "Filter Alert Level:",
                options=['Semua', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
                key="alert_sim_filter"
            )
        with sim_col2:
            n_events = st.slider(
                "Jumlah Events:",
                min_value=10,
                max_value=100,
                value=30,
                step=5,
                key="n_sim_events"
            )
        with sim_col3:
            speed = st.slider(
                "Kecepatan (detik/event):",
                min_value=0.05,
                max_value=1.0,
                value=0.15,
                step=0.05,
                key="sim_speed"
            )

        # Prepare filtered data
        sim_data = df_filtered.copy()
        if alert_filter != 'Semua':
            sim_data = sim_data[sim_data['alert_level'] == alert_filter]

        sim_data = sim_data.sort_values('event_time').head(n_events)

        # Summary counts
        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
        n_crit = len(sim_data[sim_data['alert_level'] == 'CRITICAL'])
        n_high = len(sim_data[sim_data['alert_level'] == 'HIGH'])
        n_med = len(sim_data[sim_data['alert_level'] == 'MEDIUM'])
        n_low = len(sim_data[sim_data['alert_level'] == 'LOW'])
        sum_col1.metric("🔴 Critical", n_crit)
        sum_col2.metric("🟠 High", n_high)
        sum_col3.metric("🟡 Medium", n_med)
        sum_col4.metric("🟢 Low", n_low)

        # Start simulation button
        if st.button("▶️ Mulai Simulasi Real-Time", type="primary", key="start_sim"):
            st.markdown("---")
            log_placeholder = st.empty()
            progress_bar = st.progress(0)
            log_lines = []

            for idx, (_, row) in enumerate(sim_data.iterrows()):
                event_dict = row.to_dict()
                log_entry = format_alert_log(event_dict)
                log_lines.append(log_entry)

                # Tampilkan log terbaru (max 50 baris terakhir)
                display_lines = log_lines[-50:]
                log_text = "\n\n".join(display_lines)

                log_placeholder.markdown(
                    f"""<div class="log-container"><pre style="color: #E0E0E0; margin:0; white-space: pre-wrap;">{log_text}</pre></div>""",
                    unsafe_allow_html=True
                )

                progress_bar.progress((idx + 1) / len(sim_data))
                time.sleep(speed)

            progress_bar.progress(1.0)
            st.success(f"✅ Simulasi selesai! {len(sim_data)} events ditampilkan.")
        else:
            # Tampilkan log statis saat simulasi belum dijalankan
            st.markdown("---")
            if len(sim_data) > 0:
                static_logs = []
                for _, row in sim_data.head(15).iterrows():
                    static_logs.append(format_alert_log(row.to_dict()))
                log_text = "\n\n".join(static_logs)
                st.markdown(
                    f"""<div class="log-container"><pre style="color: #E0E0E0; margin:0; white-space: pre-wrap;">{log_text}</pre></div>""",
                    unsafe_allow_html=True
                )
                st.caption(f"Menampilkan 15 dari {len(sim_data)} events. Klik tombol di atas untuk simulasi real-time.")

    # ====================================================================
    # TAB 4: DATA EXPLORER
    # ====================================================================
    with tab4:
        st.markdown("""
        <div class="section-header">
            <h3>📋 Data Explorer</h3>
        </div>
        """, unsafe_allow_html=True)

        explorer_tab1, explorer_tab2, explorer_tab3 = st.tabs([
            "📑 Events Data", "👤 Users Data", "🔧 User Features"
        ])

        with explorer_tab1:
            st.markdown(f"**Total: {len(df_filtered):,} events** (setelah filter)")
            st.dataframe(
                df_filtered.head(500),
                use_container_width=True,
                height=500,
            )

        with explorer_tab2:
            st.markdown(f"**Total: {len(df_users)} users**")
            st.dataframe(df_users, use_container_width=True, height=500)

        with explorer_tab3:
            st.markdown(f"**Total: {len(user_features)} users**")
            st.dataframe(
                user_features.style.format({
                    'failed_login_rate': '{:.4f}',
                    'avg_latency': '{:.2f}',
                    'access_to_restricted_ratio': '{:.4f}',
                    'total_bytes_out': '{:,.0f}',
                    'avg_risk_score': '{:.2f}',
                }),
                use_container_width=True,
                height=500,
            )

        # --- Terminated User Detection ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="section-header">
            <h3>⚠️ Mitigasi Risiko: Deteksi Akses Terminated User</h3>
        </div>
        """, unsafe_allow_html=True)

        if 'user_status' in df_filtered.columns:
            terminated_events = df_filtered[df_filtered['user_status'] == 'terminated']
            if len(terminated_events) > 0:
                st.error(
                    f"🚨 **PERINGATAN**: Ditemukan **{len(terminated_events)} events** "
                    f"dari **{terminated_events['user_id'].nunique()} user terminated** "
                    f"yang masih melakukan aktivitas!"
                )
                st.dataframe(
                    terminated_events[['event_time', 'user_id', 'dept', 'action',
                                      'asset_id', 'data_classification', 'status',
                                      'risk_score', 'alert_level']].head(100),
                    use_container_width=True,
                    height=300,
                )
            else:
                st.success("✅ Tidak ditemukan aktivitas dari user terminated.")

    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#78909C; padding:1rem;">
        <p style="font-size:0.85rem;">
            🛡️ <strong>Big Data Security Analytics Dashboard</strong> —
            UAS Perancangan Big Data |
            Built with Streamlit & IsolationForest |
            © 2026
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
