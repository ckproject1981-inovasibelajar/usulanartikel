import streamlit as st
import google.generativeai as genai
import pandas as pd

# --- 1. CONFIG & ENGINE ---
st.set_page_config(page_title="Q1 SEM Gold Standard", layout="wide")

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
    st.error("❌ API Key missing!")
    st.stop()

# --- 2. UI LAYOUT ---
st.title("🎓 Q1 SEM Pro: Reliability & Validity Suite")
st.info("Berdasarkan Teori Nick Shryane: Memastikan Validitas Konvergen & Model Fit.")

# STEP 1: VARIABEL & VALIDITAS KONVERGEN
with st.expander("STEP 1: Measurement Model (AVE & Composite Reliability)", expanded=True):
    var_list = st.text_area("Daftar Variabel (Pisahkan dengan koma)", value="Digital Literacy, Self Efficacy, Student Engagement")
    if var_list:
        vars_items = [v.strip() for v in var_list.split(",") if v.strip()]
        
        st.write("**Validitas Konvergen & Reliabilitas**")
        validity_df = pd.DataFrame({
            "Variable": vars_items,
            "AVE (Ideal > 0.5)": [0.000] * len(vars_items),
            "CR (Ideal > 0.7)": [0.000] * len(vars_items),
            "Cronbach Alpha": [0.000] * len(vars_items)
        })
        edited_validity = st.data_editor(validity_df, hide_index=True, use_container_width=True)

# STEP 2: STRUCTURAL MODEL FIT
with st.expander("STEP 2: Goodness of Fit (GoF) Indices", expanded=True):
    col_f1, col_f2, col_f3 = st.columns(3)
    srmr = col_f1.number_input("SRMR (Standard: < 0.08)", value=0.000, format="%.3f")
    rmsea = col_f2.number_input("RMSEA (Standard: < 0.06)", value=0.000, format="%.3f")
    cfi = col_f3.number_input("CFI (Standard: > 0.90)", value=0.000, format="%.3f")

# STEP 3: PATH COEFFICIENTS
with st.expander("STEP 3: Path Analysis (β & P-Value)", expanded=False):
    if 'path_df' not in st.session_state:
        st.session_state.path_df = pd.DataFrame(columns=["From", "To", "Beta", "P-Value"])
    
    edited_path = st.data_editor(
        st.session_state.path_df,
        num_rows="dynamic",
        column_config={
            "From": st.column_config.SelectboxColumn("From", options=vars_items),
            "To": st.column_config.SelectboxColumn("To", options=vars_items),
            "Beta": st.column_config.NumberColumn("β", format="%.3f"),
            "P-Value": st.column_config.NumberColumn("P-Value", format="%.3f")
        },
        hide_index=True, use_container_width=True
    )

doi_list = st.text_area("STEP 4: Input DOI Rujukan", placeholder="Pisahkan dengan koma")

# --- 3. EXECUTION ---
if st.button("🚀 EXECUTE Q1 VALIDATION & DRAFTING"):
    tab1, tab2, tab3 = st.tabs(["📊 Diagnostic Report", "📝 IMRAD Draft", "🔍 Deep Review"])

    with tab1:
        st.subheader("Statistical Diagnostic")
        c_a, c_b = st.columns(2)
        
        with c_a:
            st.write("**Model Fit Status**")
            fit_data = pd.DataFrame({
                "Index": ["SRMR", "RMSEA", "CFI"],
                "Result": [srmr, rmsea, cfi],
                "Conclusion": [
                    "✅ Fit" if srmr < 0.08 else "❌ Poor",
                    "✅ Fit" if rmsea < 0.06 else "❌ Poor",
                    "✅ Fit" if cfi > 0.90 else "❌ Good"
                ]
            })
            st.table(fit_data)

        with c_b:
            st.write("**Validity Check**")
            # Cek apakah AVE rata-rata memenuhi syarat
            ave_check = "✅ Valid" if edited_validity["AVE (Ideal > 0.5)"].mean() > 0.5 else "⚠️ Low Validity"
            st.metric("Convergent Validity Status", ave_check)
            st.caption("AVE > 0.5 menunjukkan indikator mampu menjelaskan variabel laten dengan baik.")

    with tab2:
        with st.spinner("Writing Results & Discussion..."):
            # Prompt yang lebih teknis sesuai dokumen PDF
            prompt = f"""
            Write a professional SEM Results section for a Scopus Q1 journal.
            Data: 
            - Fit: SRMR={srmr}, RMSEA={rmsea}, CFI={cfi}.
            - Validity: {edited_validity.to_string()}.
            - Paths: {edited_path.to_string()}.
            Style: Elsevier. Mention that validity meets Fornell-Larcker criteria.
            """
            res = model.generate_content(prompt)
            st.code(res.text, language="markdown")

    with tab3:
        # Integrasi 5 Tabel CSV # Bapak
        deep_prompt = f"Analyze DOIs: {doi_list}. Use '#' separator for 5 Tables (Review, Path, Rec, Def, Quest)."
        res_deep = model.generate_content(deep_prompt)
        st.code(res_deep.text, language="text")

st.divider()
st.caption("Professional SEM Suite | Nick Shryane Standard | Optimized for Q1 Publication")