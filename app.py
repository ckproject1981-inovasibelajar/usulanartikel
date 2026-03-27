import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
from scipy import stats
import graphviz
from docx import Document

# --- 1. SET PAGE CONFIG (Agar Antarmuka Bagus) ---
st.set_page_config(page_title="SEM Q1 PRO", layout="wide", initial_sidebar_state="expanded")

# Custom CSS untuk mempercantik UI
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #f0f2f6; border-radius: 5px 5px 0 0; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { background-color: #4CAF50 !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CORE FUNCTIONS ---

def calculate_htmt(df, prefixes):
    htmt_matrix = pd.DataFrame(index=prefixes, columns=prefixes)
    for p1 in prefixes:
        for p2 in prefixes:
            if p1 == p2:
                htmt_matrix.loc[p1, p2] = 1.0
                continue
            cols1 = [c for c in df.columns if c.startswith(p1)]
            cols2 = [c for c in df.columns if c.startswith(p2)]
            
            # Hitung korelasi antar indikator
            corrs = []
            for c1 in cols1:
                for c2 in cols2:
                    corrs.append(abs(df[c1].corr(df[c2])))
            
            # Perhitungan sederhana HTMT (Mean Hetero / Mean Mono)
            avg_hetero = np.mean(corrs)
            
            mono1 = [abs(df[c1].corr(df[c2])) for i, c1 in enumerate(cols1) for c2 in cols1[i+1:]]
            mono2 = [abs(df[c1].corr(df[c2])) for i, c1 in enumerate(cols2) for c2 in cols2[i+1:]]
            
            avg_mono = np.sqrt(np.mean(mono1 or [1]) * np.mean(mono2 or [1]))
            htmt_matrix.loc[p1, p2] = avg_hetero / (avg_mono + 1e-9)
    
    # CRITICAL FIX: Pastikan tipe data adalah float64 agar Seaborn tidak error
    return htmt_matrix.astype(float)

def run_full_analysis(df_avg, vx, vm, vy, n_iterations=1000):
    path_results, med_results, r2_values = [], [], {}
    # Gabungkan semua variabel input (X dan M) yang bisa jadi prediktor
    all_predictors = vx + vm
    targets = vm + vy
    
    for t in targets:
        # Prediktor untuk target ini adalah semua variabel input KECUALI target itu sendiri
        preds = [p for p in all_predictors if p != t]
        if not preds: continue
        
        X_o, y_o = df_avg[preds], df_avg[t]
        reg = LinearRegression().fit(X_o, y_o)
        r2_values[t] = reg.score(X_o, y_o)
        
        # Bootstrapping
        boot_c = []
        for _ in range(n_iterations):
            idx = np.random.choice(df_avg.index, len(df_avg), replace=True)
            df_b = df_avg.loc[idx]
            boot_c.append(LinearRegression().fit(df_b[preds], df_b[t]).coef_)
        
        boot_c = np.array(boot_c)
        for i, p in enumerate(preds):
            se = np.std(boot_c[:, i])
            t_stat = abs(reg.coef_[i] / (se + 1e-9))
            p_val = stats.norm.sf(t_stat) * 2
            path_results.append({
                "Path": f"{p} -> {t}", "From": p, "To": t, "Coeff": round(reg.coef_[i], 3),
                "T-Stat": round(t_stat, 3), "P-Value": round(p_val, 3),
                "Sig": "✅" if p_val < 0.05 else "❌", "R2": round(r2_values[t], 3)
            })
    
    # Analisis Mediasi Multivariabel
    for x in vx:
        for m in vm:
            for y in vy:
                # Sederhana: path a (X->M), path b (M->Y)
                try:
                    a = LinearRegression().fit(df_avg[[x]], df_avg[m]).coef_[0]
                    reg_b = LinearRegression().fit(df_avg[[m, x]], df_avg[y])
                    b = reg_b.coef_[0]
                    cp = reg_b.coef_[1] # direct effect
                    ind = a * b
                    vaf = ind / (ind + cp) if (ind + cp) != 0 else 0
                    med_results.append({
                        "Mediasi": f"{x} -> {m} -> {y}", "Indirect": round(ind, 3),
                        "VAF": f"{round(vaf*100, 1)}%", "Result": "Significant" if abs(ind) > 0.1 else "Weak"
                    })
                except: continue
                
    return pd.DataFrame(path_results), pd.DataFrame(med_results), r2_values

# --- 3. UI SIDEBAR ---

with st.sidebar:
    st.title("⚙️ Control Panel")
    uploaded_file = st.file_uploader("Upload Data (.xlsx)", type=["xlsx"])
    n_boot = st.slider("Jumlah Resample Bootstrap", 500, 5000, 1000, 500)
    st.divider()
    if st.button("Download Template Baru"):
        # Logika dummy data (sama seperti sebelumnya)
        pass

# --- 4. MAIN APP ---

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file).ffill().bfill()
    # Deteksi prefiks variabel (Contoh: X1_1 -> X1)
    prefixes = sorted(list(set([c.split('_')[0] for c in df_raw.columns if '_' in c])))
    
    with st.expander("🎯 Konfigurasi Model (Multivariabel)", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1: vx = st.multiselect("Pilih Variabel Independen (X)", prefixes)
        with col2: vm = st.multiselect("Pilih Variabel Mediator (M)", prefixes)
        with col3: vy = st.multiselect("Pilih Variabel Dependen (Y)", prefixes)

    if vx and vy:
        # Hitung rata-rata variabel
        df_avg = pd.DataFrame()
        for v in list(set(vx + vm + vy)):
            cols = [c for c in df_raw.columns if c.startswith(v)]
            df_avg[v] = df_raw[cols].mean(axis=1)

        tab1, tab2, tab3, tab4 = st.tabs(["📏 Measurement", "🏗️ Structural Model", "🧬 Mediation", "📄 Narrative"])

        # --- TAB 1: MEASUREMENT ---
        with tab1:
            st.subheader("Analisis Validitas & Distribusi")
            col_a, col_b = st.columns([1, 1.5])
            
            with col_a:
                sel = st.selectbox("Pilih Variabel untuk di-cek:", prefixes)
                fig, ax = plt.subplots(figsize=(5,3))
                sns.histplot(df_avg[sel], kde=True, color="green")
                st.pyplot(fig)
            
            with col_b:
                st.write("**HTMT Heatmap**")
                htmt_df = calculate_htmt(df_raw, prefixes)
                fig_ht, ax_ht = plt.subplots(figsize=(6, 4))
                mask = np.triu(np.ones_like(htmt_df, dtype=bool))
                sns.heatmap(htmt_df, mask=mask, annot=True, cmap="RdYlGn_r", vmin=0.5, vmax=1.0, ax=ax_ht)
                st.pyplot(fig_ht)

        # --- TAB 2: STRUCTURAL ---
        with tab2:
            if st.button("🚀 Jalankan Analisis (Bootstrapping)"):
                with st.spinner("Menghitung ribuan sampel..."):
                    p_df, m_df, r2_d = run_full_analysis(df_avg, vx, vm, vy, n_boot)
                    st.session_state.p_df = p_df
                    st.session_state.m_df = m_df
                    st.session_state.r2_d = r2_d

            if 'p_df' in st.session_state:
                p_df = st.session_state.p_df
                r2_d = st.session_state.r2_d
                
                c1, c2 = st.columns([1, 1.5])
                with c1:
                    st.write("**Path Coefficients**")
                    st.dataframe(p_df[['Path', 'Coeff', 'T-Stat', 'Sig']], use_container_width=True)
                with c2:
                    st.write("**Path Diagram**")
                    dot = graphviz.Digraph(format='png')
                    dot.attr(rankdir='LR', bgcolor='transparent')
                    for v in vx: dot.node(v, v, shape='box', color='blue')
                    for v in vm: dot.node(v, f"{v}\nR²:{round(r2_d.get(v,0),2)}", shape='ellipse', color='orange')
                    for v in vy: dot.node(v, f"{v}\nR²:{round(r2_d.get(v,0),2)}", shape='ellipse', color='green')
                    for _, row in p_df.iterrows():
                        color = "black" if row['Sig'] == '✅' else "red"
                        dot.edge(row['From'], row['To'], label=str(row['Coeff']), color=color)
                    st.graphviz_chart(dot)

        # --- TAB 3: MEDIATION ---
        with tab3:
            if 'm_df' in st.session_state:
                st.subheader("Hasil Pengaruh Tidak Langsung (Indirect Effects)")
                st.table(st.session_state.m_df)
            else:
                st.info("Jalankan analisis di tab Structural Model terlebih dahulu.")

else:
    st.info("👋 Silakan upload file Excel untuk memulai.")