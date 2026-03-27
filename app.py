import streamlit as st
import google.generativeai as genai
import pandas as pd
import re
import time
from google.api_core import exceptions

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Q1 Research Pro: Deep Reviewer", layout="wide")

# --- 2. ENGINE INITIALIZATION ---
def initialize_engine():
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        priority_list = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro']
        selected = next((t for t in priority_list if t in available_models), available_models[0] if available_models else None)
        return genai.GenerativeModel(selected), selected
    except Exception as e:
        return None, str(e)

def safe_generate(model, prompt):
    for attempt in range(3):
        try:
            return model.generate_content(prompt)
        except exceptions.ResourceExhausted:
            time.sleep(15)
        except Exception:
            break
    return None

# --- 3. SETUP API ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model_instance, active_model_name = initialize_engine()
    if not model_instance:
        st.error("Gagal memuat API.")
        st.stop()
else:
    st.error("❌ API Key missing di Streamlit Secrets!")
    st.stop()

# --- 4. UI SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Research Guard")
    st.success(f"Model: **{active_model_name}**")
    st.info("💡 Semua tabel menggunakan separator '#' tanpa spasi dan tanpa tanda petik untuk kemudahan copy-paste ke Excel.")

# --- 5. UI INPUT ---
st.title("🎓 Education AI: Scopus Q1 Deep Reviewer")

with st.expander("A. CONFIGURATION & DATA SOURCE", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        iv = st.text_input("Independent Variable(s) (X)", placeholder="e.g., Digital Literacy")
        dv = st.text_input("Dependent Variable (Y)", placeholder="e.g., Student Engagement")
        tool = st.selectbox("Alat Statistik Utama", ["PLS-SEM (SmartPLS)", "CB-SEM (AMOS)", "Multiple Regression", "ANOVA", "T-Test"])
    with col2:
        uploaded_file = st.file_uploader("Unggah Olah Data Excel (.xlsx)", type=["xlsx"])
        doi_input = st.text_area("Input Daftar DOI (Pisahkan dengan koma)", placeholder="10.1016/j.compedu.2023..., 10.1111/jcal...")

# --- 6. PROMPT BUILDER (LOGIKA EKSTRAKSI) ---
def get_extraction_prompt(dois):
    return f"""
    Based on these DOIs: {dois}. 
    Analyze each article deeply and provide 5 separate tables using '#' as separator. 
    Strict Rules: No quotes ("), No underscores, No spaces before/after #. Language: English (except translations).

    1. TABLE REVIEW: 
    Columns: DOI#Novelty#Goal#Context#Limitation#Future Recommendation#Grand Theory#Method (Quant/Qual)#Dominant Analysis#Data Collection#Country#Software#Country Type#Debates#Finding#ItemKuesioner(1/0).
    
    2. PATH HYPOTHESIS:
    Columns: FileName#DOI#iv#dv#t-test#target#jumlah responden#negara#jenis negara#konteks#theory#kesimpulan(1/0). 
    (One hypothesis per row).

    3. RECOMMENDED VARIABLES:
    Columns: doi#detail rekomendasi#nama variabel utama.
    (One variable per row).

    4. DEFINITION REVIEW:
    Columns: DOI#Variable#Definition#Dimension#Key Element#Supporting Factor#Novelty#Cronbach Alpha.

    5. QUESTIONNAIRE REVIEW:
    Columns: DOI#Variable Name#English Item#Indonesian Translation#Item Number#Loading Factor.
    """

# --- 7. EKSEKUSI ---
if st.button("🚀 START DEEP ARTICLE ANALYSIS"):
    if not doi_input:
        st.warning("Mohon masukkan daftar DOI.")
    else:
        tabs = st.tabs(["🔍 Deep Article Review", "📝 IMRAD Draft", "📊 Stats Interpretation", "📚 References"])

        with tabs[0]:
            with st.spinner("Extracting Deep Review Data..."):
                deep_prompt = get_extraction_prompt(doi_input)
                res_deep = safe_generate(model_instance, deep_prompt)
                if res_deep:
                    st.subheader("Data Extraction (CSV # Separator)")
                    st.code(res_deep.text, language="text")
                    st.download_button("Download Full Review (.txt)", res_deep.text, file_name="Deep_Review_Research.txt")

        with tabs[1]:
            with st.spinner("Generating IMRAD..."):
                imrad_prompt = f"Write a Scopus Q1 article draft. Context: {iv} to {dv} using {tool}. Use DOIs: {doi_input}. Elsevier standard, no contractions."
                res_imrad = safe_generate(model_instance, imrad_prompt)
                if res_imrad:
                    st.subheader("IMRAD Article Draft")
                    st.code(res_imrad.text, language="markdown")

        with tabs[2]:
            with st.spinner("Analyzing statistics..."):
                if uploaded_file:
                    df = pd.read_excel(uploaded_file)
                    res_stat = safe_generate(model_instance, f"Interpret this data: {df.to_string()} for {tool}.")
                    if res_stat: st.code(res_stat.text, language="text")
                else:
                    st.info("Unggah file Excel untuk melihat analisis statistik.")

        with tabs[3]:
            with st.spinner("Finalizing References..."):
                ref_res = safe_generate(model_instance, f"Format these DOIs into APA 7th: {doi_input}. No hallucinations.")
                if ref_res: st.code(ref_res.text, language="text")

st.divider()
st.caption("Anti-Hallucination & Deep Extraction Engine | Standard: Elsevier & J. Eichler")