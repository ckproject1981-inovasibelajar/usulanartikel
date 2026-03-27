import streamlit as st
import google.generativeai as genai
import pandas as pd
import time

# --- 1. CONFIG & ENGINE ---
st.set_page_config(page_title="Q1 SEM Pro: Full Analysis", layout="wide")

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
st.title("🎓 Q1 SEM Pro: Reliability, Validity & Data Interpretation")
st.info("Berdasarkan Teori Nick Shryane: Mengintegrasikan Data Mentah, Validitas Konvergen, dan Model Fit.")

# TAB UTAMA UNTUK INPUT
with st.sidebar:
    st.header("📂 Data Source")
    uploaded_raw = st.file_uploader("Unggah Data Mentah Penelitian (.xlsx)", type=["xlsx"])
    st.divider()
    st.write("**Writing Standard:** Elsevier & J. Eichler")

# STEP 1: VARIABEL & VALIDITAS KONVERGEN
with st.expander("STEP 1: Measurement Model (AVE & Composite Reliability)", expanded=True):
    var_list = st.text_area("Daftar Variabel (Pisahkan dengan koma)", value="Digital Literacy, Self Efficacy, Student Engagement")
    if var_list:
        vars_items = [v.strip() for v in var_list.split(",") if v.strip()]
        validity_df = pd.DataFrame({
            "Variable": vars_items,
            "AVE (Ideal > 0.5)": [0.550] * len(vars_items),
            "CR (Ideal > 0.7)": [0.820] * len(vars_items),
            "Cronbach Alpha": [0.780] * len(vars_items)
        })
        edited_validity = st.data_editor(validity_df, hide_index=True, use_container_width=True)

# STEP 2: STRUCTURAL MODEL FIT
with st.expander("STEP 2: Goodness of Fit (GoF) Indices", expanded=True):
    col_f1, col_f2, col_f3 = st.columns(3)
    srmr = col_f1.number_input("SRMR (Standard: < 0.08)", value=0.045, format="%.3f")
    rmsea = col_f2.number_input("RMSEA (Standard: < 0.06)", value=0.051, format="%.3f")
    cfi = col_f3.number_input("CFI (Standard: > 0.90)", value=0.940, format="%.3f")

# STEP 3: PATH COEFFICIENTS
with st.expander("STEP 3: Path Analysis (β & P-Value)", expanded=False):
    if 'path_df' not in st.session_state:
        st.session_state.path_df = pd.DataFrame([
            {"From": vars_items[0], "To": vars_items[1], "Beta": 0.450, "P-Value": 0.001},
            {"From": vars_items[1], "To": vars_items[2], "Beta": 0.380, "P-Value": 0.001}
        ])
    
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
if st.button("🚀 EXECUTE FULL Q1 ANALYSIS"):
    # Membaca data mentah jika ada
    raw_context = ""
    if uploaded_raw:
        df_raw = pd.read_excel(uploaded_raw)
        raw_context = f"Data mentah berisi {len(df_raw)} responden dengan kolom: {', '.join(df_raw.columns)}."

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Diagnostic & Fit", "📖 Interpretation", "📝 IMRAD Draft", "🔍 Deep Review"])

    with tab1:
        st.subheader("Statistical Diagnostic")
        c_a, c_b = st.columns(2)
        with c_a:
            st.write("**Model Fit Status**")
            fit_data = pd.DataFrame({
                "Index": ["SRMR", "RMSEA", "CFI"],
                "Value": [srmr, rmsea, cfi],
                "Conclusion": [
                    "✅ Fit" if srmr < 0.08 else "❌ Poor",
                    "✅ Fit" if rmsea < 0.06 else "❌ Poor",
                    "✅ Fit" if cfi > 0.90 else "❌ Good"
                ]
            })
            st.table(fit_data)
        with c_b:
            st.write("**Validity Check**")
            ave_mean = edited_validity["AVE (Ideal > 0.5)"].mean()
            st.metric("Mean AVE Score", f"{ave_mean:.3f}", delta="Valid" if ave_mean > 0.5 else "Low")
            st.caption("Berdasarkan kriteria Fornell-Larcker, AVE > 0.5 menunjukkan validitas konvergen terpenuhi.")

    with tab2:
        st.subheader("💡 Data Interpretation")
        with st.spinner("Analyzing data patterns..."):
            interpret_prompt = f"""
            Sebagai ahli statistik SEM, berikan interpretasi mendalam dalam Bahasa Indonesia untuk:
            1. Data Mentah: {raw_context}
            2. Validitas: {edited_validity.to_string()}
            3. Model Fit: SRMR {srmr}, RMSEA {rmsea}, CFI {cfi}.
            4. Jalur: {edited_path.to_string()}
            Jelaskan hubungan antar variabel dan apakah hipotesis diterima. 
            Gunakan gaya bahasa akademik untuk naskah publikasi.
            """
            inter_res = model.generate_content(interpret_prompt)
            st.write(inter_res.text)

    with tab3:
        with st.spinner("Writing IMRAD Results..."):
            prompt = f"""
            Write a Scopus Q1 Results and Discussion section. 
            Integrate Model Fit ({srmr}, {rmsea}, {cfi}) and Path Coefficients.
            Context: {raw_context}. Variables: {var_list}. DOIs: {doi_list}.
            Include theoretical implications based on the data. Style: Elsevier.
            """
            res = model.generate_content(prompt)
            st.code(res.text, language="markdown")

    with tab4:
        with st.spinner("Running 5-Table Deep Review..."):
            deep_prompt = f"Analyze DOIs: {doi_list}. Generate 5 Tables (Review, Path, Rec, Def, Quest) using '#' separator."
            res_deep = model.generate_content(deep_prompt)
            st.code(res_deep.text, language="text")

st.divider()
st.caption("Professional SEM Suite | Nick Shryane Standard | Developed by Citra Kurniawan - 2026")