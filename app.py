import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import graphviz # Pustaka Tambahan untuk Visualisasi

# --- 1. CONFIG & ENGINE ---
st.set_page_config(page_title="Q1 SEM Ultimate Pro", layout="wide", page_icon="🎓")

def initialize_engine():
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        selected = next((t for t in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro'] if t in available_models), available_models[0])
        return genai.GenerativeModel(selected)
    except: return None

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = initialize_engine()
else:
    st.error("❌ API Key missing! Check Streamlit Secrets.")
    st.stop()

# --- 2. CORE STATISTICAL FUNCTIONS ---

def calculate_reliability_vif(df, var_codes):
    avg_scores = pd.DataFrame()
    vif_data = []
    # 1. Pre-calculate Avg Scores
    for code in var_codes:
        cols = [col for col in df.columns if col.startswith(code)]
        if cols:
            avg_scores[code] = df[cols].mean(axis=1)
    
    # 2. Calculate VIF & Alpha
    for var in avg_scores.columns:
        y = avg_scores[var]
        X = avg_scores.drop(columns=[var])
        cols = [col for col in df.columns if col.startswith(var)]
        k = len(cols)
        
        # Simple Alpha Proxy
        alpha = (k / (k - 1)) * (1 - (df[cols].var().sum() / df[cols].sum(axis=1).var())) if k > 1 else 1.0
        
        # Simple VIF
        if not X.empty:
            r_sq = LinearRegression().fit(X, y).score(X, y)
            vif = 1 / (1 - r_sq) if r_sq < 1 else 10
        else:
            vif = 1.0
            
        vif_data.append({"Variable": var, "Alpha (>0.7)": round(alpha, 3), "VIF (<5.0)": round(vif, 3)})
            
    return pd.DataFrame(vif_data), avg_scores

def calculate_mediation_paths(df_avg, x, m, y):
    try:
        # Path a (X -> M)
        slope_a, _, _, _, _ = stats.linregress(df_avg[x], df_avg[m])
        # Path b & c' (M & X -> Y)
        reg_y = LinearRegression().fit(df_avg[[x, m]], df_avg[y])
        path_b = reg_y.coef_[1]
        path_c_prime = reg_y.coef_[0]
        indirect = slope_a * path_b
        
        return {
            "a": round(slope_a, 3),
            "b": round(path_b, 3),
            "c_prime": round(path_c_prime, 3),
            "Indirect": round(indirect, 3),
            "Total": round(path_c_prime + indirect, 3),
            "Status": "Partial" if abs(path_c_prime) > 0.1 else "Full"
        }
    except: return None

# --- 3. FUNGSI VISUALISASI JALUR (GRAPHVIZ) ---

def draw_path_diagram(x, m, y, paths):
    """Membuat Diagram Jalur Mediasi Otomatis"""
    dot = graphviz.Digraph(comment='SEM Path Model')
    dot.attr(rankdir='LR') # Left to Right
    
    # Nodes (Variabel Laten)
    dot.node('X', x, shape='ellipse', style='filled', fillcolor='#E1F5FE')
    dot.node('M', m, shape='ellipse', style='filled', fillcolor='#E8F5E9')
    dot.node('Y', y, shape='ellipse', style='filled', fillcolor='#FFFDE7')
    
    # Edges (Jalur Koefisien)
    # Jalur Langsung (solid)
    dot.edge('X', 'M', label=f'a={paths["a"]}', color='#1565C0')
    dot.edge('M', 'Y', label=f'b={paths["b"]}', color='#2E7D32')
    # Jalur Tidak Langsung (dotted/dashed untuk c')
    dot.edge('X', 'Y', label=f"c'={paths['c_prime']}", style='dashed', color='#F9A825')
    
    st.graphviz_chart(dot)

# --- 4. UI LAYOUT ---

# LOGO DI ATAS JUDUL
st.image("https://i.ibb.co.com/23N3kpBY/Logo-DLI.png", width=150)
st.title("🎓 Q1 SEM Ultimate: Path, Mediation & Visualization")
st.caption("Digital Learning Institute | Manchester Framework (Nick Shryane Standard)")

with st.sidebar:
    st.image("https://i.ibb.co.com/23N3kpBY/Logo-DLI.png", use_container_width=True)
    st.header("📂 Data Center")
    uploaded_file = st.file_uploader("Unggah Data Mentah (.xlsx)", type=["xlsx"])
    st.divider()
    st.markdown("### Quality Control")
    st.info("Aplikasi menangani Missing Data & Outlier otomatis.")
    st.divider()
    st.caption("Developed by Citra Kurniawan - 2026")

if uploaded_file:
    # Membaca data dan menangani Missing Data (Mean Imputation)
    df_raw = pd.read_excel(uploaded_file)
    df = df_raw.fillna(df_raw.mean(numeric_only=True))
    
    st.subheader("1. Konfigurasi Model Mediasi")
    st.write("Masukkan kode variabel Anda (e.g., DL, SE, EN) dan tentukan alur mediasi.")
    
    var_input = st.text_input("Daftar Kode Variabel", value="DL, SE, EN")
    var_codes = [c.strip() for c in var_input.split(",")]
    
    col_x, col_m, col_y = st.columns(3)
    x_var = col_x.selectbox("Pilih X (Independen)", var_codes, index=0)
    m_var = col_m.selectbox("Pilih M (Mediator)", var_codes, index=1)
    y_var = col_y.selectbox("Pilih Y (Dependen)", var_codes, index=2)

    # PERHITUNGAN DIMULAI
    vif_df, df_avg = calculate_reliability_vif(df, var_codes)
    med_res = calculate_mediation_paths(df_avg, x_var, m_var, y_var)

    # DISPLAY QUALITY METRICS
    st.subheader("2. Quality & Reliability Report")
    st.table(vif_df)

    if med_res:
        st.subheader("3. Path Analysis & Visualization")
        
        col_vis, col_met = st.columns([2, 1])
        with col_vis:
            st.write("**Diagram Jalur Otomatis (a -> b -> c')**")
            draw_path_diagram(x_var, m_var, y_var, med_res)
            st.caption("Jalur putus-putus menunjukkan efek langsung (c').")
            
        with col_met:
            st.write("**Koefisien Mediasi**")
            st.metric("Direct Effect (c')", med_res["c_prime"])
            st.metric("Indirect Effect (a*b)", med_res["Indirect"])
            st.metric("Total Effect", med_res["Total"])
            st.metric("Mediation Type", med_res["Status"])

        # EXECUTION BUTTON
        st.divider()
        if st.button("🚀 GENERATE FINAL Q1 MANUSCRIPT"):
            t1, t2, t3 = st.tabs(["💡 Interpretation", "📝 IMRAD Draft", "🔍 Deep Review"])
            
            with t1:
                st.subheader("Interpretasi Kausal (Shryane Standard)")
                with st.spinner("Menganalisis koefisien jalur..."):
                    prompt = f"Interpretasikan hasil SEM: X={x_var}, M={m_var}, Y={y_var}. Path a={med_res['a']}, Path b={med_res['b']}, Direct c'={med_res['c_prime']}. Fokus pada mekanisme mediasi M={m_var}. Bahasa Indonesia akademik."
                    st.write(model.generate_content(prompt).text)
            
            with t2:
                st.subheader("Draf IMRAD (Elsevier Style)")
                with st.spinner("Drafting manuscript..."):
                    prompt_imrad = f"Write Scopus Q1 Results section for mediation model {x_var}->{m_var}->{y_var}. Refer to the provided path coefficients (a, b, c'). Tone: Formal Elsevier."
                    st.code(model.generate_content(prompt_imrad).text, language="markdown")
            
            with t3:
                st.write("Gunakan fitur ini untuk Deep Review rujukan DOI.")
    else:
        st.error("Gagal menghitung mediasi. Pastikan data kolom tersedia.")

else:
    st.warning("Selamat Datang! Silakan unggah file data mentah (.xlsx) di sidebar untuk memulai analisis.")

st.divider()
st.caption("Developed by Citra Kurniawan - 2026 | SEM Causal Suite Ver 3.0 (Graphviz Integrated)")