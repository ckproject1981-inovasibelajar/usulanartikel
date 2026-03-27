import streamlit as st
import google.generativeai as genai
import pandas as pd
import re

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Q1 Research Pro: End-to-End Suite", layout="wide")

# --- 2. FUNGSI LOGIKA AI (STABILIZER) ---
def get_best_model():
    try:
        # Mengambil daftar model yang tersedia untuk API Key tersebut
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Daftar model target dengan urutan prioritas (dengan prefix models/)
        target_models = [
            'models/gemini-1.5-pro', 
            'models/gemini-1.5-flash', 
            'models/gemini-pro'
        ]
        
        # Mencari kecocokan antara model yang tersedia dan model target
        for target in target_models:
            if target in available_models:
                return target
        
        # Fallback jika tidak ada yang cocok
        return available_models[0] if available_models else "models/gemini-1.5-flash"
    except Exception:
        # Fallback hardcoded jika list_models gagal (umum di beberapa environment)
        return "models/gemini-1.5-flash"

# --- 3. KONEKSI KE API VIA SECRETS ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    # Inisialisasi model secara global agar konsisten
    SELECTED_MODEL_NAME = get_best_model()
    model_instance = genai.GenerativeModel(SELECTED_MODEL_NAME)
else:
    st.error("❌ API Key missing! Harap masukkan GEMINI_API_KEY di Streamlit Secrets.")
    st.stop()

# --- 4. MODUL ACADEMIC GUARD & FORMATTER ---
def check_academic_consistency(text):
    colloquialisms = {
        r"\bcan't\b": "cannot", r"\bdon't\b": "do not", r"\bisn't\b": "is not",
        r"\baren't\b": "are not", r"\bget\b": "obtain", r"\bdone\b": "conducted",
        r"\bobviously\b": "clearly", r"\ba lot of\b": "numerous"
    }
    warnings = []
    for pattern, replacement in colloquialisms.items():
        if re.search(pattern, text, re.IGNORECASE):
            warnings.append(f"⚠️ Gunakan '{replacement}' alih-alih '{pattern.strip(r'|b')}'.")
    return warnings

# --- 5. UI SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Research Engine")
    st.info(f"Model Aktif: **{SELECTED_MODEL_NAME}**")
    with st.expander("ℹ️ PETUNJUK PENYEMPURNAAN", expanded=True):
        st.markdown("""
        * **Multi-Variabel:** Support input X, M, Y jamak (koma).
        * **Kategori Lengkap:** Mencakup SEM, Komparatif, hingga Non-Parametrik.
        * **Grammar Guard:** Deteksi otomatis gaya bahasa informal (Elsevier Standard).
        * **Reference Formatter:** Otomasi APA/Harvard dari DOI.
        """)
    st.write("**Standards:** Elsevier & J. Eichler")

# --- 6. UI INPUT ---
st.title("🎓 Education AI: Scopus Q1 End-to-End Builder")

with st.expander("A. STATISTICAL & VARIABLE SETUP", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        iv = st.text_input("Independent Variable(s) (X)", placeholder="e.g., Digital Literacy, ICT Skills")
        mv = st.text_input("Mediator/Moderator (M/Z)", placeholder="e.g., Teacher Self-Efficacy")
        dv = st.text_input("Dependent Variable (Y)", placeholder="e.g., Student Engagement")
    with col2:
        category = st.selectbox("Kategori Analisis", ["1. Komparatif", "2. Asosiatif", "3. SEM", "4. Deskriptif", "5. Non-Parametrik"])
        tool = st.selectbox("Alat Statistik Spesifik", ["PLS-SEM (SmartPLS)", "CB-SEM (AMOS)", "Multiple Regression", "ANOVA", "T-Test", "Mann-Whitney"])

with st.expander("B. RESEARCH DATA & REFERENCES"):
    col_a, col_b = st.columns(2)
    with col_a:
        uploaded_file = st.file_uploader("Unggah Hasil Olah Data (.xlsx)", type=["xlsx"])
        data_str = ""
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.dataframe(df, height=150)
            data_str = df.to_string()
    with col_b:
        ref_style = st.radio("Pilih Format Referensi", ["APA 7th Edition", "Harvard"], horizontal=True)
        doi_list = st.text_area("Input DOI (Pisahkan dengan koma)", key="doi_input")

# --- 7. EKSEKUSI ---
if st.button("🚀 EXECUTE FULL RESEARCH SUITE"):
    if not doi_list:
        st.warning("Input DOI diperlukan untuk analisis Gap dan Referensi.")
    else:
        context = f"Vars: X={iv}, M={mv}, Y={dv} | Tool: {tool} | Data: {data_str} | DOI: {doi_list}"
        
        tabs = st.tabs(["📊 Statistical Analysis", "📝 IMRAD Draft", "🛡️ Grammar Audit", "📚 Reference List"])

        # TAB 1: Analisis Statistik
        with tabs[0]:
            with st.spinner("Analyzing stats..."):
                try:
                    res_stat = model_instance.generate_content(f"Provide professional interpretation for {tool} based on: {data_str}. Focus on p-values and significance. Academic English.")
                    st.markdown(res_stat.text)
                except Exception as e:
                    st.error(f"Error pada Analisis: {e}")

        # TAB 2: Draf Artikel
        with tabs[1]:
            with st.spinner("Drafting IMRAD..."):
                imrad_prompt = f"Write a Scopus Q1 Education article draft. Context: {context}. Include Research Gap from DOIs and Hypotheses (H1, H2, etc). Elsevier Style (Active voice, no contractions)."
                res_imrad = model_instance.generate_content(imrad_prompt)
                st.session_state['current_draft'] = res_imrad.text
                st.markdown(res_imrad.text)

        # TAB 3: Audit Grammar
        with tabs[2]:
            st.subheader("🛡️ Elsevier Academic Audit")
            if 'current_draft' in st.session_state:
                warnings = check_academic_consistency(st.session_state['current_draft'])
                if warnings:
                    for w in warnings: st.warning(w)
                    if st.button("Auto-Refine Draft"):
                        refined = model_instance.generate_content(f"Refine this text to 100% Elsevier Standard (No contractions, no colloquialism): {st.session_state['current_draft']}")
                        st.markdown(refined.text)
                else: st.success("✅ Clean of colloquialisms!")

        # TAB 4: Reference Formatter
        with tabs[3]:
            with st.spinner(f"Formatting references in {ref_style}..."):
                ref_prompt = f"Based on these DOIs: {doi_list}, generate a formal Reference List in {ref_style} format. Alphabetical order. Academic style."
                res_refs = model_instance.generate_content(ref_prompt)
                st.markdown(f"### References ({ref_style})")
                st.code(res_refs.text, language="text")
                st.download_button("Download References (.txt)", res_refs.text, file_name="References.txt")

st.divider()
st.caption("End-to-End Q1 Research Suite | Statistical & Bibliographic Integrity")