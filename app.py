import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. CONFIGURATION & LOADING ---
@st.cache_resource
def load_assets():
    model_dir = "model"
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    scaler_amount = joblib.load(os.path.join(model_dir, "scaler_amount.pkl"))
    scaler_time = joblib.load(os.path.join(model_dir, "scaler_time.pkl"))
    
    # Optional assets depending on how the model was trained
    try:
        selected_features = joblib.load(os.path.join(model_dir, "selected_features.pkl"))
    except:
        selected_features = None
        
    try:
        feature_names = joblib.load(os.path.join(model_dir, "feature_names.pkl"))
    except:
        feature_names = None
        
    return model, scaler_amount, scaler_time, selected_features, feature_names

@st.cache_data
def load_sample_data():
    # Load sample dataset (karena creditcard.csv asli 150MB tidak diunggah ke Git)
    df = pd.read_csv("sample_creditcard.csv")
    return df

try:
    model, scaler_amount, scaler_time, selected_features, feature_names = load_assets()
    df_pool = load_sample_data()
except Exception as e:
    st.error(f"Error loading assets or data: {e}")
    st.stop()

# --- 2. SESSION STATE SETUP ---
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Tx ID', 'Time (s)', 'Amount ($)', 'Status', 'Prediction'])
if 'tx_counter' not in st.session_state:
    st.session_state.tx_counter = 0

# --- 3. UI LAYOUT ---
st.title("🛡️ Sistem Deteksi Penipuan Transaksi (Real-time Simulation)")
st.markdown("Memantau transaksi kartu kredit dan mendeteksi anomali menggunakan model Machine Learning yang telah dilatih dengan metode SMOTE + Genetic Algorithm.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Panel Kontrol")
    st.markdown("Gunakan panel ini untuk mensimulasikan transaksi yang masuk ke sistem.")
    
    manual_sim = st.button("Simulasikan 1 Transaksi", use_container_width=True, type="primary")
    auto_sim = st.toggle("Simulasi Otomatis (Real-time)")
    
    if manual_sim or auto_sim:
        # Ambil 1 baris berurutan
        current_idx = st.session_state.tx_counter % len(df_pool)
        raw_tx = df_pool.iloc[[current_idx]].copy()
        actual_class = raw_tx['Class'].values[0]
        
        # Simpan nilai asli untuk UI
        raw_amount = raw_tx['Amount'].values[0]
        raw_time = raw_tx['Time'].values[0]
        
        # Preprocessing (seperti di backend)
        raw_tx_features = raw_tx.drop('Class', axis=1)
        raw_tx_features['Amount'] = scaler_amount.transform(raw_tx_features[['Amount']])
        raw_tx_features['Time'] = scaler_time.transform(raw_tx_features[['Time']])
        
        # Rename agar sesuai dengan format model saat training
        raw_tx_features = raw_tx_features.rename(columns={'Amount': 'Amount_scaled', 'Time': 'Time_scaled'})
        
        # Reorder kolom agar urutannya persis sama dengan saat training (sangat penting!)
        if feature_names is not None:
            raw_tx_features = raw_tx_features[feature_names]

        # Filter fitur jika menggunakan GA
        if selected_features is not None:
            # Jika selected_features adalah array boolean atau index
            X_input = raw_tx_features.values[:, selected_features]
        else:
            X_input = raw_tx_features.values
            
        # Prediksi
        pred = model.predict(X_input)[0]
        
        # Tambahkan ke history
        new_row = pd.DataFrame([{
            'Tx ID': f"TXN-{1000 + st.session_state.tx_counter}",
            'Time (s)': f"{raw_time:.0f}",
            'Amount ($)': f"${raw_amount:,.2f}",
            'Status': 'Aman' if actual_class == 0 else 'Fraud (Aktual)',
            'Prediction': 'Aman' if pred == 0 else 'FRAUD (Terdeteksi)'
        }])
        
        st.session_state.history = pd.concat([new_row, st.session_state.history], ignore_index=True).head(100)
        st.session_state.tx_counter += 1
        st.session_state.last_pred = pred
        st.session_state.last_amount = raw_amount

    if st.button("Bersihkan Riwayat", use_container_width=True):
        st.session_state.history = pd.DataFrame(columns=['Tx ID', 'Time (s)', 'Amount ($)', 'Status', 'Prediction'])
        st.session_state.tx_counter = 0
        if 'last_pred' in st.session_state:
            del st.session_state.last_pred

    st.divider()
    st.markdown("💡 **Info Model:**")
    st.info(f"Model: {type(model).__name__}")
    if selected_features is not None:
        st.info(f"Fitur Terpilih: {sum(selected_features if isinstance(selected_features[0], (bool, np.bool_)) else [1]*len(selected_features))} dari {len(df_pool.columns) - 1}")

# --- MAIN DASHBOARD ---
col1, col2, col3 = st.columns(3)

total_tx = len(st.session_state.history)
total_fraud_detected = len(st.session_state.history[st.session_state.history['Prediction'] == 'FRAUD (Terdeteksi)'])

with col1:
    st.metric(label="Total Transaksi Diproses", value=total_tx)
with col2:
    st.metric(label="Fraud Terdeteksi (Model)", value=total_fraud_detected)
with col3:
    if total_tx > 0:
        fraud_rate = (total_fraud_detected / total_tx) * 100
        st.metric(label="Persentase Fraud", value=f"{fraud_rate:.1f}%")
    else:
        st.metric(label="Persentase Fraud", value="0%")

st.divider()

# --- ALERT SYSTEM ---
if 'last_pred' in st.session_state:
    st.subheader("📡 Status Transaksi Terakhir")
    if st.session_state.last_pred == 1:
        st.error(f"🚨 ALARM! Transaksi sebesar **${st.session_state.last_amount:,.2f}** DIBLOKIR. Terindikasi sebagai Penipuan (Fraud).")
    else:
        st.success(f"✅ Transaksi sebesar **${st.session_state.last_amount:,.2f}** BERHASIL. Aman dari indikasi Fraud.")
else:
    st.info("Sistem siap sedia. Menunggu transaksi masuk...")

# --- TRANSACTION HISTORY ---
st.subheader("📋 Riwayat Transaksi (Real-time)")
if not st.session_state.history.empty:
    # Color formatting function for dataframe
    def color_status(val):
        color = '#ff4b4b' if 'FRAUD' in val else '#21c354' if 'Aman' in val else ''
        return f'color: {color}; font-weight: bold;'
    
    st.dataframe(
        st.session_state.history.style.map(color_status, subset=['Prediction']),
        use_container_width=True,
        hide_index=True
    )
else:
    st.write("Belum ada transaksi.")

# Trigger simulasi kontinu jika auto_sim menyala
if auto_sim:
    time.sleep(1.5)
    st.rerun()

