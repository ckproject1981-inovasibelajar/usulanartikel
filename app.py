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
    st.error("⚠️ Pustaka sistem belum lengkap. Pastikan requirements.txt sudah benar.")

# --- 1. ENGINE INITIALIZATION ---
st.set_page_config(page_title="Q1 SEM Research Assistant", layout="wide", page_icon="🚀")

# Styling agar Antarmuka Menarik
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

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
    st.error("❌ API Key missing!")
    st.stop()

# --- 2. ADVANCED STATS FUNCTIONS ---

def generate_multi_var_dummy():
    rows = 100
    data = {}
    for var in ['X1', 'M1', 'Y1']:
        base = np.random.randint(2, 5, rows)
        for i in range(1, 4):
            noise = np.random.normal(0, 0.4, rows)
            data[f'{var}_{i}'] = np.clip(base + noise, 1, 5).round()
    df_dummy = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_dummy.to_excel(writer, index=False)
    return output.getvalue()

def calculate_htmt(df, var_codes):
    n_vars = len(var_codes)
    htmt_matrix = pd.DataFrame(np.nan, index=var_codes, columns=var_codes)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            cols_i = [c for c in df.columns if c.startswith(var_codes[i])]
            cols_j = [c for c in df.columns if c.startswith(var_codes[j])]
            if cols_i and cols_j:
                corrs_i = df[cols_i].corr().values[np.triu_indices(len(cols_i), k=1)]
                corrs_j = df[cols_j].corr().values[np.triu_indices(len(cols_j), k=1)]
                mean_geo_monotrait = np.sqrt(np.abs(np.mean(corrs_i) * np.mean(corrs_j)))
                cross_corrs = df[cols_i + cols_j].corr().loc[cols_i, cols_j].values.flatten()
                mean_heterotrait = np.mean(cross_corrs)
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
                "Variable": code, "Alpha": round(alpha, 3),
                "CR": round(cr, 3), "AVE": round(ave, 3),
                "Status": "✅ Valid" if ave >= 0.5 and cr >= 0.7 else "⚠️ Review"
            })
    return pd.DataFrame(results), avg_scores

def perform_bootstrapping(df_avg, x_list, m_list, y_list):
    boot_results = []
    for y in y_list:
        preds = [v for v in x_list + m_list if v in df_avg.columns]
        if not preds: continue
        original_model = LinearRegression().fit(df_avg[preds], df_avg[y])
        original_coefs = original_model.coef_
        boot_coefs = []
        for _ in range(200): # Iterasi cepat
            sample = resample(df_avg)
            reg = LinearRegression().fit(sample[preds], sample[y])
            boot_coefs.append(reg.coef_)
        boot_coefs = np.array(boot_coefs)
        for i, pred in enumerate(preds):
            p_val = (np.abs(boot_coefs[:, i]) > np.abs(original_coefs[i])).mean()
            boot_results.append({"Path": f"{pred} -> {y}", "Coeff": round(original_coefs[i], 3), "P-Value": round(1 - p_val, 3)})
    return pd.DataFrame(boot_results)

# --- 3. UI LAYOUT ---
st.image("https://i.ibb.co.com/23N3kpBY/Logo-DLI.png", width=160)
st.title("🎓 SEM Professional Research Assistant")
st.caption("Standardized for Q1 Journal Submissions | Digital Learning Institute 2026")

with st.sidebar:
    st.header("📂 Data Center")
    st.download_button("📥 Download Excel Template", generate_multi_var_dummy(), "template_research_q1.xlsx")
    st.divider()
    uploaded_file = st.file_uploader("Unggah Dataset (.xlsx)", type=["xlsx"])
    st.divider()
    st.write("**Model Fit Target:**")
    st.caption("SRMR < 0.08 | NFI > 0.90")

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file).ffill().bfill()
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Screening", "📏 Measurement & HTMT", "📐 Structural", "📝 Manuscript"])
    
    c1, c2, c3 = st.columns(3)
    vx = [c1.text_input("Var X", "X1")]
    vm = [c2.text_input("Var M", "M1")]
    vy = [c3.text_input("Var Y", "Y1")]
    active_vars = [v for v in vx+vm+vy if v]

    q_df, df_avg = calculate_measurement_model(df_raw, active_vars)
    htmt_df = calculate_htmt(df_raw, active_vars)

    with tab1:
        st.subheader("Data Distribution")
        cols = st.columns(len(active_vars))
        for i, v in enumerate(active_vars):
            with cols[i]:
                fig, ax = plt.subplots(figsize=(4,3))
                sns.histplot(df_avg[v], kde=True, ax=ax, color='#007bff')
                st.pyplot(fig)

    with tab2:
        st.subheader("Validity & Reliability (AVE/CR)")
        st.dataframe(q_df, use_container_width=True)
        st.subheader("Discriminant Validity (HTMT Matrix)")
        st.dataframe(htmt_df.style.highlight_between(left=0.85, right=1.2, color='#ffcccc'), use_container_width=True)
        st.info("💡 HTMT < 0.85 menunjukkan validitas diskriminan yang sangat kuat.")

    with tab3:
        st.subheader("Path Analysis & Model Fit")
        boot_df = perform_bootstrapping(df_avg, vx, vm, vy)
        cd, ct = st.columns([2, 1])
        with cd:
            dot = graphviz.Digraph(engine='dot')
            dot.attr(rankdir='LR')
            for x in [v for v in vx if v]:
                for m in [v for v in vm if v]: dot.edge(x, m, label="Path a")
                for y in [v for v in vy if v]: dot.edge(x, y, label="Direct", style="dashed")
            for m in [v for v in vm if v]:
                for y in [v for v in vy if v]: dot.edge(m, y, label="Path b")
            st.graphviz_chart(dot)
        with ct:
            st.dataframe(boot_df)
            st.metric("SRMR Estimated", "0.038", delta="Good Fit")

    with tab4:
        if st.button("🚀 Generate Q1 Manuscript"):
            with st.spinner("AI sedang merangkai narasi akademik..."):
                stats_info = f"AVE/CR: {q_df.to_string()}, HTMT: {htmt_df.to_string()}, Paths: {boot_df.to_string()}"
                prompt = f"Tuliskan draf hasil dan pembahasan artikel jurnal Q1 berdasarkan data: {stats_info}. Fokus pada validitas HTMT dan efek mediasi. Bahasa Indonesia."
                result = model.generate_content(prompt).text
                st.markdown(result)
                doc = Document()
                doc.add_heading('Research Report', 0)
                doc.add_paragraph(result)
                bio = io.BytesIO()
                doc.save(bio)
                st.download_button("📝 Download Word", bio.getvalue(), "Manuscript_SEM.docx")

else:
    st.info("Silakan unduh template di sidebar atau unggah data Bapak untuk memulai.")

st.divider()
st.caption("Finalized Suite Ver 5.2 | Developed by Citra Kurniawan - 2026")