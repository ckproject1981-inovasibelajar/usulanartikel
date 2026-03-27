import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import io
try:
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    import graphviz
except ImportError:
    st.error("⚠️ Pustaka sistem belum lengkap. Pastikan file requirements.txt & packages.txt sudah ada.")

# --- 1. CONFIG & ENGINE (REPAIRED) ---
st.set_page_config(page_title="Q1 SEM Multi-Variable Pro", layout="wide", page_icon="🎓")

def initialize_engine():
    """Fungsi untuk mencari model yang tersedia secara otomatis guna menghindari NotFound error"""
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Prioritas: Flash 1.5, lalu Pro 1.5, lalu apa saja yang tersedia
        selected = next((t for t in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro'] if t in available_models), available_models[0])
        return genai.GenerativeModel(selected)
    except Exception as e:
        st.error(f"Gagal inisialisasi AI: {e}")
        return None

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = initialize_engine()
else:
    st.error("❌ API Key Gemini tidak ditemukan di Streamlit Secrets.")
    st.stop()

# --- 2. GENERATE DUMMY DATA ---
def generate_multi_var_dummy():
    rows = 70
    data = {}
    for i in range(1, 6):
        base = np.random.randint(2, 5, rows)
        for j in range(1, 4): 
            data[f'X{i}_{j}'] = np.clip(base + np.random.normal(0, 0.5, rows), 1, 5).round()
            data[f'M{i}_{j}'] = np.clip(base * 0.5 + np.random.normal(0, 0.7, rows), 1, 5).round()
            data[f'Y{i}_{j}'] = np.clip(base * 0.4 + np.random.normal(0, 0.8, rows), 1, 5).round()
    df_dummy = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_dummy.to_excel(writer, index=False)
    return output.getvalue()

# --- 3. ANALYTICS FUNCTION ---
def analyze_multi_path(df, x_list, m_list, y_list):
    all_vars = x_list + m_list + y_list
    avg_scores = pd.DataFrame()
    q_reports = []
    
    for var in all_vars:
        if var:
            cols = [c for c in df.columns if c.startswith(var)]
            if cols:
                avg_scores[var] = df[cols].mean(axis=1)
                k = len(cols)
                alpha = (k/(k-1)) * (1-(df[cols].var().sum()/df[cols].sum(axis=1).var())) if k > 1 else 1.0
                q_reports.append({"Variable": var, "Alpha": round(alpha, 3)})

    for var in avg_scores.columns:
        target = avg_scores[var]
        feat = avg_scores.drop(columns=[var])
        if not feat.empty:
            r2 = LinearRegression().fit(feat, target).score(feat, target)
            vif = 1/(1-r2) if r2 < 1 else 10
            for item in q_reports:
                if item["Variable"] == var: item["VIF"] = round(vif, 3)
    return pd.DataFrame(q_reports), avg_scores

# --- 4. UI LAYOUT ---
st.image("https://i.ibb.co.com/23N3kpBY/Logo-DLI.png", width=160)
st.title("🎓 Q1 SEM Ultimate: Multi-Path Suite")

with st.sidebar:
    st.image("https://i.ibb.co.com/23N3kpBY/Logo-DLI.png", use_container_width=True)
    st.header("📂 Data Center")
    st.download_button("📥 Download Multi-Var Template", generate_multi_var_dummy(), "template_sem.xlsx")
    st.divider()
    uploaded_file = st.file_uploader("Unggah Data Riset (.xlsx)", type=["xlsx"])
    st.caption("Standard: Digital Learning Institute - 2026")

if uploaded_file:
    df = pd.read_excel(uploaded_file).ffill().bfill()
    
    st.subheader("1. Konfigurasi Multi-Variabel")
    col_x, col_m, col_y = st.columns(3)
    active_x = [col_x.text_input(f"X{i}", "X1" if i==1 else "") for i in range(1, 6)]
    active_m = [col_m.text_input(f"M{i}", "M1" if i==1 else "") for i in range(1, 6)]
    active_y = [col_y.text_input(f"Y{i}", "Y1" if i==1 else "") for i in range(1, 6)]
    
    active_x = [x for x in active_x if x]
    active_m = [m for m in active_m if m]
    active_y = [y for y in active_y if y]
    
    if active_x and active_y:
        q_df, df_avg = analyze_multi_path(df, active_x, active_m, active_y)
        st.subheader("2. Report Kualitas Konstruk")
        st.dataframe(q_df, use_container_width=True)
        
        st.subheader("3. Visualisasi Jalur (Structural Model)")
        dot = graphviz.Digraph(engine='dot')
        dot.attr(rankdir='LR', nodesep='0.4', ranksep='1.2')
        
        # Nodes & Edges Logic
        for x in active_x: dot.node(x, x, shape='ellipse', style='filled', color='#E1F5FE')
        for m in active_m: dot.node(m, m, shape='ellipse', style='filled', color='#E8F5E9')
        for y in active_y: dot.node(y, y, shape='ellipse', style='filled', color='#FFFDE7')
        
        for y in active_y:
            predictors = active_x + active_m
            reg = LinearRegression().fit(df_avg[predictors], df_avg[y])
            coefs = dict(zip(predictors, reg.coef_))
            for m in active_m: dot.edge(m, y, label=f"b={round(coefs[m],2)}", color='#2E7D32')
            for x in active_x: dot.edge(x, y, label=f"c'={round(coefs[x],2)}", style='dashed', color='#F9A825')
        
        for x in active_x:
            for m in active_m:
                slope, _, _, _, _ = stats.linregress(df_avg[x], df_avg[m])
                dot.edge(x, m, label=f"a={round(slope,2)}", color='#1565C0')

        st.graphviz_chart(dot)

        if st.button("🚀 GENERATE MANUSCRIPT"):
            if model:
                with st.spinner("Menghubungi AI..."):
                    try:
                        summary_stats = df_avg.corr().to_string()
                        prompt = f"Berikan interpretasi profesional hasil SEM: X:{active_x}, M:{active_m}, Y:{active_y}. Gunakan korelasi: {summary_stats}. Fokus pada efek langsung vs tidak langsung. Bahasa Indonesia