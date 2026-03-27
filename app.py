import streamlit as st
import google.generativeai as genai
import pandas as pd

# --- 1. KONFIGURASI ---
st.set_page_config(page_title="Q1 Research Pro: Excel Analyzer", layout="wide")

def get_best_model():
    try:
        models = genai.list_models()
        valid_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        preferred = ['gemini-1.5-pro', 'gemini-1.5-flash']
        for p in preferred:
            match = next((m for m in valid_models if p in m), None)
            if match: return match
        return "models/gemini-1.5-flash"
    except: return "models/gemini-1.5-flash"

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model_instance = genai.GenerativeModel(get_best_model())
else:
    st.error("❌ API Key missing!")
    st.stop()

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Statistical Engine")
    st.write("**Standards:** Elsevier & Eichler")
    st.markdown("---")
    st.info("Fitur ini akan membaca tabel hasil olah data (SPSS/SmartPLS) Anda untuk menyusun narasi Result.")

# --- 3. UI INPUT ---
st.title("🎓 Education AI: Excel Data Analyst")

with st.expander("A. STATISTICAL CONFIGURATION", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        iv = st.text_input("Independent Variable", key="iv")
        dv = st.text_input("Dependent Variable", key="dv")
    with col2:
        tool = st.selectbox("Alat Statistik", ["Multiple Linear Regression", "PLS-SEM (SmartPLS)", "CB-SEM (AMOS)", "T-test/ANOVA"])

# --- 4. PENYEMPURNAAN RESEARCH DATA (EXCEL UPLOAD) ---
with st.expander("B. RESEARCH DATA INPUT (EXCEL)", expanded=True):
    st.write("Unggah file Excel berisi tabel hasil olah data (Coefficients, Path Coefficients, atau Reliability).")
    uploaded_file = st.file_uploader("Pilih file .xlsx", type=["xlsx"])
    
    extracted_data_string = ""
    if uploaded_file:
        try:
            # Membaca semua sheet untuk fleksibilitas
            df = pd.read_excel(uploaded_file)
            st.success("Data berhasil dibaca!")
            st.dataframe(df, height=200) # Menampilkan preview ke user
            
            # Mengonversi dataframe menjadi string agar bisa dibaca AI
            extracted_data_string = df.to_string(index=False)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")

    doi_list = st.text_area("C. Upload DOI Pendukung", key="doi_input")

# --- 5. EKSEKUSI ---
if st.button("🚀 ANALISIS DATA & GENERATE ARTIKEL"):
    if not extracted_data_string or not doi_list:
        st.warning("Mohon unggah file Excel dan masukkan DOI.")
    else:
        context = f"""
        Method: {tool}. 
        Variables: {iv} to {dv}.
        Statistical Data from Excel:
        {extracted_data_string}
        Reference DOIs: {doi_list}
        """
        
        tabs = st.tabs(["Statistical Analysis", "Full Article Draft"])
        
        with tabs[0]:
            with st.spinner("Menganalisis angka statistik..."):
                # Prompt khusus untuk bedah angka
                stat_prompt = f"As a statistician, interpret this data table: {extracted_data_string}. Focus on significance (p-values), effect size, and hypothesis support for {tool}. Output in academic English."
                res_stat = model_instance.generate_content(stat_prompt)
                st.markdown(res_stat.text)
        
        with tabs[1]:
            with st.spinner("Menyusun draf Q1..."):
                # Prompt IMRAD berdasarkan Elsevier & Eichler
                imrad_prompt = f"Write a professional Scopus Q1 article draft using this data: {context}. Ensure the 'Result' section precisely mentions the numbers from the table. Use Elsevier scientific writing style (active voice, no contractions)."
                res_imrad = model_instance.generate_content(imrad_prompt)
                st.markdown(res_imrad.text)