import streamlit as st
import google.generativeai as genai
import pandas as pd
import re
import time
from google.api_core import exceptions

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Q1 Research Pro: Ultra-Stable", layout="wide")

# --- 2. FUNGSI LOGIKA MODEL (ANTI-ERROR 404 & 429) ---
def initialize_engine():
    """Menginisialisasi model dengan validasi keberadaan model di API"""
    try:
        # Ambil daftar semua model yang mendukung generateContent
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Daftar prioritas (mencoba dengan dan tanpa prefix untuk kompatibilitas)
        priority_list = [
            'models/gemini-1.5-flash', 
            'models/gemini-1.5-pro',
            'gemini-1.5-flash', 
            'gemini-1.5-pro'
        ]
        
        selected = None
        for target in priority_list:
            if target in available_models:
                selected = target
                break
        
        # Jika tidak ada yang cocok di list, ambil apa saja yang tersedia pertama kali
        if not selected and available_models:
            selected = available_models[0]
            
        if not selected:
            return None, "Tidak ada model Gemini yang ditemukan pada API Key ini."
            
        return genai.GenerativeModel(selected), selected
    except Exception as e:
        return None, str(e)

def safe_generate(model, prompt):
    """Fungsi eksekusi dengan Retry Logic untuk mengatasi ResourceExhausted (429)"""
    for attempt in range(3):
        try:
            return model.generate_content(prompt)
        except exceptions.ResourceExhausted:
            st.warning(f"Kuota penuh. Menunggu 12 detik... (Percobaan {attempt+1}/3)")
            time.sleep(12)
        except Exception as e:
            st.error(f"Kesalahan API: {e}")
            break
    return None

# --- 3. SETUP API ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model_instance, active_model_name = initialize_engine()
    if not model_instance:
        st.error(f"Gagal Inisialisasi: {active_model_name}")
        st.stop()
else:
    st.error("❌ API Key tidak ditemukan di Streamlit Secrets!")
    st.stop()

# --- 4. MODUL ACADEMIC GUARD ---
def check_academic_consistency(text):
    colloquialisms = {
        r"\bcan't\b": "cannot", r"\bdon't\b": "do not", r"\bisn't\b": "is not",
        r"\bget\b": "obtain", r"\bdone\b": "conducted", r"\ba lot of\b": "numerous"
    }
    warnings = []
    for pattern, replacement in colloquialisms.items():
        if re.search(pattern, text, re.IGNORECASE):
            warnings.append(f"⚠️ Gunakan '{replacement}' alih-alih '{pattern.strip(r'|b')}'.")
    return warnings

# --- 5. UI LAYOUT ---
st.title("🎓 Education AI: Scopus Q1 End-to-End Builder")

with st.sidebar:
    st.title("🛡️ Research Engine")
    st.success(f"Model Aktif: **{active_model_name}**")
    st.markdown("""
    **Fitur Terpasang:**
    - Auto-Model Discovery (Fix 404)
    - Rate-Limit Handling (Fix 429)
    - Elsevier Grammar Guard
    - APA/Harvard Formatter
    """)

# A. Input Variabel & Statistik
with st.expander("A. STATISTICAL & VARIABLE SETUP", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        iv = st.text_input("Independent Variable(s) (X)", placeholder="e.g., Digital Literacy, ICT Skills")
        mv = st.text_input("Mediator/Moderator (M/Z)", placeholder="e.g., Teacher Self-Efficacy")
        dv = st.text_input("Dependent Variable (Y)", placeholder="e.g., Student Engagement")
    with col2:
        tool = st.selectbox("Alat Statistik Spesifik", ["PLS-SEM (SmartPLS)", "CB-SEM (AMOS)", "Multiple Regression", "ANOVA", "T-Test"])

# B. Data Excel & DOI
with st.expander("B. DATA SOURCE & REFERENCES"):
    col_a, col_b = st.columns(2)
    with col_a:
        uploaded_file = st.file_uploader("Unggah Hasil Olah Data (.xlsx)", type=["xlsx"])
        data_str = ""
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.dataframe(df, height=150)
            data_str = df.to_string()
    with col_b:
        ref_style = st.radio("Format Referensi", ["APA 7th Edition", "Harvard"], horizontal=True)
        doi_list = st.text_area("Input DOI (Pisahkan dengan koma)", key="doi_input")

# --- 6. EKSEKUSI ---
if st.button("🚀 EXECUTE FULL RESEARCH SUITE"):
    if not doi_list:
        st.warning("Mohon masukkan DOI pendukung.")
    else:
        context = f"Vars: X={iv}, M={mv}, Y={dv} | Tool: {tool} | Data: {data_str} | DOI: {doi_list}"
        tabs = st.tabs(["📊 Statistics", "📝 IMRAD Draft", "🛡️ Grammar Audit", "📚 References"])

        with tabs[0]:
            with st.spinner("Analyzing stats..."):
                res = safe_generate(model_instance, f"Interpret these statistics professionally: {data_str} using {tool}.")
                if res: st.markdown(res.text)

        with tabs[1]:
            with st.spinner("Drafting Q1 Article..."):
                prompt = f"Write a Scopus Q1 article draft based on Elsevier/Eichler standards. Context: {context}. Include Gap & Hypotheses."
                res = safe_generate(model_instance, prompt)
                if res: 
                    st.session_state['current_draft'] = res.text
                    st.markdown(res.text)

        with tabs[2]:
            st.subheader("🛡️ Academic Integrity Check")
            if 'current_draft' in st.session_state:
                warnings = check_academic_consistency(st.session_state['current_draft'])
                if warnings:
                    for w in warnings: st.warning(w)
                else: st.success("✅ Naskah bebas dari colloquialism sesuai standar Elsevier.")

        with tabs[3]:
            with st.spinner("Formatting references..."):
                res = safe_generate(model_instance, f"Format these DOIs into {ref_style} Reference List: {doi_list}.")
                if res: st.code(res.text)

st.divider()
st.caption("Auto-Inisialisasi Aktif | Kompatibel dengan Tier Gratis & Berbayar")