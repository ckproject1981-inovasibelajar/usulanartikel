import streamlit as st
import google.generativeai as genai
import pandas as pd
import re
import time
from google.api_core import exceptions

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Q1 Research Pro: Rate-Limit Safe", layout="wide")

# --- 2. FUNGSI GENERATE DENGAN RETRY LOGIC (MENGATASI QUOTA EXCEEDED) ---
def safe_generate_content(model, prompt):
    max_retries = 3
    for i in range(max_retries):
        try:
            return model.generate_content(prompt)
        except exceptions.ResourceExhausted:
            if i < max_retries - 1:
                st.warning(f"Quota penuh. Menunggu 10 detik sebelum mencoba lagi (Percobaan {i+1}/{max_retries})...")
                time.sleep(10) # Jeda untuk reset kuota
            else:
                st.error("❌ Kuota API Anda benar-benar habis. Silakan tunggu 1-2 menit atau gunakan API Key lain.")
                return None
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
            return None

def get_best_model():
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        target_models = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro']
        for target in target_models:
            if target in available_models: return target
        return "models/gemini-1.5-flash"
    except: return "models/gemini-1.5-flash"

# --- 3. KONEKSI API ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model_name = get_best_model()
    model_instance = genai.GenerativeModel(model_name)
else:
    st.error("❌ API Key missing!")
    st.stop()

# --- 4. UI & INPUT ---
st.title("🎓 Education AI: Scopus Q1 End-to-End Builder")

with st.sidebar:
    st.title("🛡️ Research Engine")
    st.info(f"Model: **{model_name}**")
    st.markdown("""
    **Fitur Utama:**
    - Anti-Error Rate Limit
    - Multi-Variable (X, M, Y)
    - Elsevier Grammar Guard
    - APA/Harvard Formatter
    """)

with st.expander("A. SETTING VARIABEL & STATISTIK", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        iv = st.text_input("Independent Variable(s) (X)", placeholder="e.g., Digital Literacy, ICT Skills")
        mv = st.text_input("Mediator/Moderator (M/Z)", placeholder="e.g., Teacher Self-Efficacy")
        dv = st.text_input("Dependent Variable (Y)", placeholder="e.g., Student Engagement")
    with col2:
        category = st.selectbox("Kategori Analisis", ["1. Komparatif", "2. Asosiatif", "3. SEM", "4. Deskriptif", "5. Non-Parametrik"])
        tool = st.selectbox("Alat Statistik Spesifik", ["PLS-SEM (SmartPLS)", "CB-SEM (AMOS)", "Multiple Regression", "ANOVA", "T-Test"])

with st.expander("B. DATA EXCEL & DOI"):
    col_a, col_b = st.columns(2)
    with col_a:
        uploaded_file = st.file_uploader("Unggah Hasil (.xlsx)", type=["xlsx"])
        data_str = ""
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.dataframe(df, height=150)
            data_str = df.to_string()
    with col_b:
        ref_style = st.radio("Format Referensi", ["APA 7th Edition", "Harvard"], horizontal=True)
        doi_list = st.text_area("Input DOI (Gunakan koma) (5-10 DOI)", key="doi_input")

# --- 5. EKSEKUSI ---
if st.button("🚀 EXECUTE FULL RESEARCH SUITE"):
    if not doi_list:
        st.warning("Input DOI diperlukan.")
    else:
        context = f"Vars: X={iv}, M={mv}, Y={dv} | Tool: {tool} | Data: {data_str} | DOI: {doi_list}"
        tabs = st.tabs(["📊 Stats Analysis", "📝 IMRAD Draft", "📚 References"])

        with tabs[0]:
            with st.spinner("Analyzing stats..."):
                res = safe_generate_content(model_instance, f"Interpret this data: {data_str} for {tool}.")
                if res: st.markdown(res.text)

        with tabs[1]:
            with st.spinner("Drafting IMRAD..."):
                prompt = f"Write a Scopus Q1 article draft based on Elsevier standards. Context: {context}. Include Research Gap & Hypotheses."
                res = safe_generate_content(model_instance, prompt)
                if res: st.markdown(res.text)

        with tabs[2]:
            with st.spinner("Formatting references..."):
                prompt = f"Format these DOIs into {ref_style}: {doi_list}."
                res = safe_generate_content(model_instance, prompt)
                if res: st.code(res.text)

st.divider()
st.caption("Rate-Limit Handling Active | Elsevier Academic Standards")