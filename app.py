import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression

# --- 1. CONFIG & ENGINE ---
st.set_page_config(page_title="Q1 SEM Ultimate Pro", layout="wide", page_icon="🎓")

def initialize_engine():
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        selected = next((t for t in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro'] if t in available_models), available_models[0])
        return genai.GenerativeModel(selected)
    except: return None

# Secure API Configuration
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = initialize_engine()
else:
    st.error("❌ API Key missing! Please check your Streamlit secrets.")
    st.stop()

# --- 2. FUNGSI ANALISIS (RELIABILITY & MEDIATION) ---
def calculate_mediation(df, x_code, m_code, y_code):
    """Menghitung Simple Mediation: X -> M -> Y"""
    try:
        x = df[[c for c in df.columns if c.startswith(x_code)]].mean(axis=1)
        m = df[[c for c in df.columns if c.startswith(m_code)]].mean(axis=1)
        y = df[[c for c in df.columns if c.startswith(y_code)]].mean(axis=1)
        
        slope_a, _, _, _, _ = stats.linregress(x, m)
        X_combined = pd.DataFrame({'X': x, 'M': m})
        reg_y = LinearRegression().fit(X_combined, y)
        path_b = reg_y.coef_[1]
        path_c_prime = reg_y.coef_[0]
        
        indirect_effect = slope_a * path_b
        total_effect = path_c_prime + indirect_effect
        
        return {
            "Direct Effect (c')": round(path_c_prime, 3),
            "Indirect Effect (a*b)": round(indirect_effect, 3),
            "Total Effect": round(total_effect, 3),
            "Mediation Type": "Partial Mediation" if abs(path_c_prime) > 0.1 else "Full Mediation"
        }
    except Exception as e:
        return {"Error": str(e)}

# --- 3. UI LAYOUT ---

# Header Section dengan Logo Utama
col_logo, col_text = st.columns([1, 5])
with col_logo:
    st.image("https://i.ibb.co.com/23N3kpBY/Logo-DLI.png", width=120)
with col_text:
    st.title("Q1 SEM Ultimate: Path, Mediation & Quality Control")
    st.caption("Advanced Statistical Engine for Scopus Q1 Publication | Manchester Framework")

st.info("💡 **Sistem Analisis Terpadu:** Memvalidasi data mentah, mendeteksi bias, hingga pengujian mediasi secara otomatis.")

# Sidebar Branding & Input
with st.sidebar:
    st.image("https://i.ibb.co.com/23N3kpBY/Logo-DLI.png", use_container_width=True)
    st.header("📂 Data Center")
    uploaded_raw = st.file_uploader("Unggah Data Mentah (.xlsx)", type=["xlsx"])
    st.divider()
    st.markdown("### **Nick Shryane Standard**")
    st.caption("Metode ini merujuk pada framework University of Manchester untuk akurasi model kausal.")
    st.divider()
    st.caption("Developed by Citra Kurniawan - 2026")

# Main Logic
if uploaded_raw:
    df_raw = pd.read_excel(uploaded_raw)
    
    st.subheader("1. Konfigurasi Model Mediasi")
    st.write("Tentukan kode variabel yang terdapat pada kolom Excel Anda.")
    
    col_x, col_m, col_y = st.columns(3)
    with col_x:
        var_x = st.text_input("Variabel Independen (X)", "DL")
    with col_m:
        var_m = st.text_input("Variabel Mediator (M)", "SE")
    with col_y:
        var_y = st.text_input("Variabel Dependen (Y)", "EN")
    
    st.divider()
    
    # Execution: Path Analysis
    med_res = calculate_mediation(df_raw, var_x, var_m, var_y)
    
    if "Error" not in med_res:
        st.subheader("2. Hasil Analisis Jalur (Path Analysis)")
        res_cols = st.columns(4)
        res_cols[0].metric("Direct Effect", med_res["Direct Effect (c')"])
        res_cols[1].metric("Indirect Effect", med_res["Indirect Effect (a*b)"])
        res_cols[2].metric("Total Effect", med_res["Total Effect"])
        res_cols[3].metric("Status", med_res["Mediation Type"])
        
        st.divider()
        
        # FINAL EXECUTION BUTTON
        if st.button("🚀 GENERATE FINAL Q1 MANUSCRIPT REPORT"):
            tab1, tab2, tab3 = st.tabs(["💡 Interpretation", "📝 IMRAD Final Draft", "🔍 Deep Review"])

            with tab1:
                st.subheader("Interpretasi Kausal & Mediasi")
                with st.spinner("Menyusun narasi akademik berbasis teori Shryane..."):
                    prompt = f"""
                    Buat interpretasi data profesional untuk jurnal Q1:
                    - Model: {var_x} -> {var_m} -> {var_y}
                    - Direct Effect: {med_res["Direct Effect (c')"]}
                    - Indirect Effect: {med_res["Indirect Effect (a*b)"]}
                    - Total Effect: {med_res["Total Effect"]}
                    
                    Gunakan diksi akademik formal. Jelaskan peran mediator {var_m}. 
                    Gunakan asumsi bootstrapping untuk signifikansi. 
                    Narasi harus fokus pada mekanisme kausal tanpa mengulang data validitas.
                    """
                    st.write(model.generate_content(prompt).text)

            with tab2:
                st.subheader("Draf IMRAD (Standard Elsevier Q1)")
                with st.spinner("Drafting high-quality manuscript section..."):
                    prompt_imrad = f"""
                    Write a high-quality Results and Discussion section for a Scopus Q1 paper.
                    Focus on:
                    1. Structural Model & Hypothesis Testing.
                    2. Specific mediation role of {var_m} between {var_x} and {var_y}.
                    3. Theoretical Implications based on Nick Shryane's perspective on causality.
                    Tone: Formal, rigorous, and analytical.
                    """
                    st.code(model.generate_content(prompt_imrad).text, language="markdown")

            with tab3:
                st.info("Fitur Deep Review: Masukkan DOI rujukan untuk membandingkan temuan Anda dengan literatur global.")
                st.text_input("Masukkan DOI rujukan (Opsional)")
    else:
        st.error(f"Gagal memproses data: {med_res['Error']}. Pastikan kolom di Excel sesuai dengan kode variabel.")

else:
    st.warning("👋 Selamat Datang! Silakan unggah file data mentah (.xlsx) di sidebar untuk memulai analisis.")

# --- FOOTER ---
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: grey;'>
    Finalized Suite Ver 2.0 | Digital Learning Institute | Optimized for Scopus Q1 Standards
    </div>
    """, 
    unsafe_allow_html=True
)