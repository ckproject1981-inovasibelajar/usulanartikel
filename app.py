import streamlit as st
import google.generativeai as genai

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Q1 Research Pro: Full IMRAD Generator", layout="wide")

# --- 2. FUNGSI OTOMATISASI MODEL GEMINI ---
def get_best_model():
    try:
        models = genai.list_models()
        valid_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        preferred = ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro']
        for p in preferred:
            match = next((m for m in valid_models if p in m), None)
            if match: return match
        return valid_models[0] if valid_models else "models/gemini-1.5-flash"
    except:
        return "models/gemini-1.5-flash"

# --- 3. KONEKSI KE API VIA SECRETS ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    SELECTED_MODEL = get_best_model()
    model_instance = genai.GenerativeModel(SELECTED_MODEL)
else:
    st.error("❌ GEMINI_API_KEY tidak ditemukan di Streamlit Secrets!")
    st.stop()

# --- 4. MASTER PROMPTS (Integrasi Panduan Elsevier) ---
# Prompt ini telah dimodifikasi untuk mematuhi kaidah penulisan world-class paper
PROMPT_MASTER = {
    "Review": """Cari kebaruan penelitian, tujuan penelitian, konteks penelitian, keterbatasan penelitian, rekomendasi penelitian kedepan dan grand theory yang digunakan... [Prompt Review Sebelumnya]""",
    "Path Hipotesis": """Identifikasi apakah ada hipotesis... [Prompt Path Sebelumnya]""",
    "Recommended Variables": """Identifikasi Future recommendation... [Prompt Rec Sebelumnya]""",
    "Definition Review": """Pelajari semua bagian artikel... [Prompt Definition Sebelumnya]""",
    "Questionnaire Review": """Ekstraksi kuesioner... [Prompt Questionnaire Sebelumnya]"""
}

# --- 5. SIDEBAR & TOKEN COUNTER ---
with st.sidebar:
    st.title("🛡️ Research Engine")
    st.info(f"🤖 Model: **{SELECTED_MODEL}**")
    
    st.subheader("📊 Token Monitoring")
    input_text = f"{st.session_state.get('doi_input', '')} {st.session_state.get('iv', '')} {st.session_state.get('dv', '')}"
    try:
        token_count = model_instance.count_tokens(input_text).total_tokens
        st.metric("Estimated Tokens", f"{token_count:,}")
        max_tokens = 1000000 if "1.5" in SELECTED_MODEL else 32768
        st.progress(min(token_count / max_tokens, 1.0))
    except:
        st.caption("Waiting for input...")
    
    st.markdown("---")
    st.write("**Standards Applied:** Elsevier Author Workshop & Joerg Eichler Guidelines")

# --- 6. UI LAYOUT ---
st.title("🎓 Education AI: Scopus Q1 Full Article Builder")

col_a, col_b = st.columns(2)
with col_a:
    iv = st.text_input("Independent Variable", key="iv")
    mv = st.text_input("Moderator/Mediator Variable", key="mv")
    dv = st.text_input("Dependent Variable", key="dv")
with col_b:
    tool = st.text_input("Statistical Tool (e.g., SmartPLS, AMOS)", value="SmartPLS 4", key="tool")
    doi_list = st.text_area("Input DOIs (Comma separated)", key="doi_input")

# --- 7. PROSES EKSEKUSI ---
if st.button("🚀 GENERATE FULL PROFESSIONAL ANALYSIS"):
    if not doi_list:
        st.warning("Mohon masukkan DOI terlebih dahulu.")
    else:
        context = f"Variables: {iv} (IV), {mv} (MV), {dv} (DV). Method: Quantitative with {tool}. DOIs: {doi_list}."
        
        tab_titles = list(PROMPT_MASTER.keys()) + ["Full Article Draft"]
        tabs = st.tabs(tab_titles)

        # Tab 1-5 tetap menggunakan format ekstraksi tabel #
        for i, (name, prompt) in enumerate(PROMPT_MASTER.items()):
            with tabs[i]:
                with st.spinner(f"Processing {name}..."):
                    try:
                        res = model_instance.generate_content(f"{prompt}\n\nDATA:\n{context}")
                        st.code(res.text, language="text")
                        st.download_button(f"Download {name} CSV", res.text, file_name=f"{name.lower().replace(' ', '_')}.csv")
                    except Exception as e:
                        st.error(f"Error processing {name}: {e}")

        # Tab 6: Full Article Draft (Mengikuti Panduan Elsevier/Eichler)
        with tabs[-1]:
            with st.spinner("Drafting Full Q1 Article (Elsevier Standards)..."):
                imrad_full_prompt = f"""
                As a world-class academic editor familiar with Elsevier and Jörg Eichler's publishing standards, draft a research paper in English.
                
                STRICT ADHERENCE TO RULES:
                - Use active voice where possible.
                - No colloquialisms, no contractions (e.g., use 'do not' instead of 'don't').
                - Be concise and clear.
                - Ensure a logical flow from Title to Conclusion.

                Parameters: IV: {iv}, MV: {mv}, DV: {dv}, Tool: {tool}. 
                Context DOIs: {doi_list}.

                REQUIRED STRUCTURE:
                1. ABSTRACT: Background, Purpose, Method, Result, and Value. Concise (250 words max).
                2. INTRODUCTION: The 'Why'. Establish the gap, cite significance, and state objectives.
                3. METHOD: The 'How'. Explain sampling, instruments, and {tool} analytical steps.
                4. RESULT: Presentation of data without interpretation.
                5. ANALYSIS: Deep interpretation of coefficients, R-squared, and p-values.
                6. DISCUSSION: The 'So What'. Compare findings with provided DOIs. Highlight theoretical/practical implications.
                7. CONCLUSION: Final takeaway, study limitations, and future outlook.

                Style: High-impact Academic English (Scopus Q1).
                """
                full_res = model_instance.generate_content(imrad_full_prompt)
                st.markdown(full_res.text)
                st.download_button("Download Full Draft (.txt)", full_res.text, file_name="Q1_Article_Draft.txt")

st.divider()
st.info("💡 **Academic Note:** Output ini telah disesuaikan dengan panduan penulisan Elsevier (Clear, Concise, Correct).")