import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from docx import Document
try:
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    from sklearn.utils import resample
    import graphviz
except ImportError:
    st.error("⚠️ Pustaka sistem belum lengkap.")

# --- 1. ENGINE INITIALIZATION ---
st.set_page_config(page_title="Q1 SEM Research Assistant", layout="wide", page_icon="🚀")

def initialize_engine():
    try:
        available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        selected = next((t for t in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro'] if t in available), available[0])
        return genai.GenerativeModel(selected)
    except: return None

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = initialize_engine()
else:
    st.stop()

# --- 2. ADVANCED STATS FUNCTIONS (HTMT ADDED) ---

def calculate_htmt(df, var_codes):
    """Menghitung matriks HTMT antar variabel laten"""
    n_vars = len(var_codes)
    htmt_matrix = pd.DataFrame(np.nan, index=var_codes, columns=var_codes)
    
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            cols_i = [c for c in df.columns if c.startswith(var_codes[i])]
            cols_j = [c for c in df.columns if c.startswith(var_codes[j])]
            
            if cols_i and cols_j:
                # 1. Mean korelasi antar-item dalam variabel yang sama (Monotrait-Heteromethod)
                corrs_i = df[cols_i].corr().values[np.triu_indices(len(cols_i), k=1)]
                corrs_j = df[cols_j].corr().values[np.triu_indices(len(cols_j), k=1)]
                
                mean_geo_monotrait = np.sqrt(np.mean(corrs_i) * np.mean(corrs_j))
                
                # 2. Mean korelasi antar-item lintas variabel (Heterotrait-Heteromethod)
                cross_corrs = df[cols_i + cols_j].corr().loc[cols_i, cols_j].values.flatten()
                mean_heterotrait = np.mean(cross_corrs)
                
                # 3. Rasio HTMT
                htmt_val = mean_heterotrait / mean_geo_monotrait if mean_geo_monotrait != 0 else 0
                htmt_matrix.iloc[j, i] = round(htmt_val, 3)
                
    return htmt_matrix

def calculate_measurement_model(df, var_codes):
    results = []
    avg_scores = pd.DataFrame()
    for code in var_codes:
        cols = [c for c in df.columns if c.startswith(code)]
        if cols:
            loadings = [stats.pearsonr(df[col], df[cols].mean(axis=1))[0] for col in cols]
            ave = sum([l**2 for l in loadings]) / len(cols)
            cr = (sum(loadings)**2) / (sum(loadings)**2 + sum([1 - l**2 for l in loadings]))
            k = len(cols)
            alpha = (k/(k-1)) * (1-(df[cols].var().sum()/df[cols].sum(axis=1).var()))
            avg_scores[code] = df[cols].mean(axis=1)
            results.append({
                "Variable": code, "Cronbach Alpha": round(alpha, 3),
                "CR": round(cr, 3), "AVE": round(ave, 3)
            })
    return pd.DataFrame(results), avg_scores

# --- 3. UI LAYOUT ---
st.title("🎓 SEM Professional Suite: HTMT & Discriminant Validity")

with st.sidebar:
    uploaded_file = st.file_uploader("Unggah Dataset (.xlsx)", type=["xlsx"])
    st.info("💡 HTMT < 0.85 (Strict) atau < 0.90 (Liberal) menunjukkan validitas diskriminan yang baik.")

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file).ffill().bfill()
    tab1, tab2, tab3 = st.tabs(["📏 Measurement & HTMT", "📐 Structural Model", "📝 Manuscript"])
    
    c1, c2, c3 = st.columns(3)
    vx = [c1.text_input("Var X", "X1")]
    vm = [c2.text_input("Var M", "M1")]
    vy = [c3.text_input("Var Y", "Y1")]
    active_vars = [v for v in vx+vm+vy if v]

    q_df, df_avg = calculate_measurement_model(df_raw, active_vars)
    htmt_df = calculate_htmt(df_raw, active_vars)

    with tab1:
        st.subheader("Validitas Konvergen (AVE/CR)")
        st.table(q_df)
        
        st.subheader("Validitas Diskriminan (HTMT Matrix)")
        st.dataframe(htmt_df.style.highlight_between(left=0.85, right=1.0, color='#ffcccc'))
        st.caption("Nilai berwarna merah menunjukkan potensi tumpang tindih antar variabel (HTMT > 0.85).")

    with tab2:
        # (Fungsi visualisasi diagram tetap seperti versi sebelumnya)
        dot = graphviz.Digraph(engine='dot')
        dot.attr(rankdir='LR')
        for x in vx:
            for m in vm: dot.edge(x, m, label="a")
            for y in vy: dot.edge(x, y, label="c'", style="dashed")
        for m in vm:
            for y in vy: dot.edge(m, y, label="b")
        st.graphviz_chart(dot)

    with tab3:
        if st.button("🚀 Generate Journal-Ready Manuscript"):
            prompt = f"Tuliskan bab hasil penelitian SEM. Sertakan analisis validitas diskriminan menggunakan HTMT dengan data: {htmt_df.to_string()}. Jika nilai < 0.85, nyatakan validitas diskriminan terpenuhi. Bahasa Indonesia akademik."
            result = model.generate_content(prompt).text
            st.markdown(result)