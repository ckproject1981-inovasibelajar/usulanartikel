import streamlit as st
import google.generativeai as genai
import pandas as pd
import re
import time
from google.api_core import exceptions

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Q1 Research Pro: Copy Ready", layout="wide")

# --- 2. ENGINE INITIALIZATION (ANTI-ERROR 404) ---
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
            time.sleep(12)
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
    st.error("❌ API Key missing!")
    st.stop()

# --- 4. UI SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Research Guard")
    st.success(f"Active: **{active_model_name}**")
    st.info("💡 Gunakan ikon di pojok kanan atas kotak teks untuk menyalin draf langsung ke MS Word.")
    st.write("**Writing Standard:** Elsevier & J. Eichler")

# --- 5. UI INPUT ---
st.title("🎓 Education AI: Scopus Q1 Full Suite")

with st.expander("A. SETTING VARIABEL & STATISTIK", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        iv = st.text_input("Independent Variable(s) (X)", placeholder="e.g., Digital Literacy, ICT Skills")
        mv = st.text_input("Mediator (M)", placeholder="e.g., Self-Efficacy")
        dv = st.text_input("Dependent Variable (Y)", placeholder="e.g., Student Engagement")
    with col2:
        tool = st.selectbox("Alat Statistik", ["PLS-SEM (SmartPLS)", "CB-SEM (AMOS)", "Multiple Regression", "ANOVA", "T-Test"])

with st.expander("B. DATA SOURCE & REFERENCES"):
    uploaded_file = st.file_uploader("Unggah Hasil (.xlsx)", type=["xlsx"])
    data_str = ""
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        data_str = df.to_string()
    
    ref_style = st.radio("Format Referensi", ["APA 7th Edition", "Harvard"], horizontal=True)
    doi_list = st.text_area("Input DOI (Pisahkan dengan koma)", key="doi_input")

# --- 6. EKSEKUSI ---
if st.button("🚀 EXECUTE FULL RESEARCH SUITE"):
    if not doi_list:
        st.warning("Mohon masukkan DOI rujukan.")
    else:
        context = f"Vars: X={iv}, M={mv}, Y={dv} | Tool: {tool} | Data: {data_str} | DOI: {doi_list}"
        tabs = st.tabs(["📊 Statistics", "📝 IMRAD Draft", "📚 Verified References"])

        with tabs[0]:
            with st.spinner("Analyzing stats..."):
                res = safe_generate(model_instance, f"Interpret these statistics: {data_str} for {tool}.")
                if res:
                    st.subheader("Statistical Interpretation")
                    st.code(res.text, language="markdown") # Menggunakan st.code agar bisa di-copy

        with tabs[1]:
            with st.spinner("Drafting Q1 Article..."):
                prompt = f"""
                Write a Scopus Q1 article draft. Context: {context}.
                STRICT RULE: Only use provided variables and DOIs. No hallucination.
                Style: Elsevier Standard (Active voice, no contractions).
                """
                res = safe_generate(model_instance, prompt)
                if res:
                    st.subheader("IMRAD Article Draft")
                    # Fitur salin otomatis via st.code
                    st.code(res.text, language="markdown")
                    st.download_button("Download Draft (.txt)", res.text, file_name="Draft_Artikel_Q1.txt")

        with tabs[2]:
            with st.spinner("Formatting references..."):
                ref_prompt = f"""
                Format these DOIs into {ref_style}: {doi_list}.
                STRICT: DO NOT HALLUCINATE. If DOI is unknown, return [VERIFY DOI: xxx].
                Sort alphabetically.
                """
                res = safe_generate(model_instance, ref_prompt)
                if res:
                    st.subheader(f"Reference List ({ref_style})")
                    # Fitur salin otomatis via st.code
                    st.code(res.text, language="text")
                    st.download_button("Download References (.txt)", res.text, file_name="References.txt")

st.divider()
st.caption("Anti-Hallucination Engine | Powered by Gemini 3 Flash")