import streamlit as st
import google.generativeai as genai
import pandas as pd
import time

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Q1 Research Pro: Adaptive", layout="wide")

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
        except Exception:
            time.sleep(10)
    return None

# --- 3. SETUP API ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model_instance, active_model_name = initialize_engine()
else:
    st.error("❌ API Key missing! Silakan cek Streamlit Secrets.")
    st.stop()

# --- 4. UI INPUT & DYNAMIC LOGIC ---
st.title("🎓 Education AI: Scopus Q1 Adaptive Suite")

with st.expander("A. STATISTICAL CONFIGURATION", expanded=True):
    # PILIHAN ANALISIS DULU
    tool = st.selectbox("Pilih Jenis Analisis", 
                        ["PLS-SEM (SmartPLS)", "CB-SEM (AMOS)", "Multiple Linear Regression", "T-Test / ANOVA"])
    
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Variabel Input")
        # Logika Kondisional Variabel
        iv = st.text_input("Independent Variable (X)", value="Digital Literacy")
        
        # Mediator hanya muncul jika SEM
        mv = None
        if "SEM" in tool:
            mv = st.text_input("Mediator Variable (M)", value="Teacher Self-Efficacy")
        
        dv = st.text_input("Dependent Variable (Y)", value="Student Engagement")

    with col2:
        st.subheader("Koefisien Jalur (β) / Statistik")
        # Logika Kondisional Input Koefisien
        beta_xm, beta_my, beta_xy = 0.0, 0.0, 0.0
        
        if "SEM" in tool:
            c1, c2, c3 = st.columns(3)
            beta_xm = c1.number_input("β (X→M)", value=0.45)
            beta_my = c2.number_input("β (M→Y)", value=0.38)
            beta_xy = c3.number_input("β (X→Y) Direct", value=0.12)
        elif "Regression" in tool:
            beta_xy = st.number_input("Standardized Beta (β)", value=0.50)
        else: # T-Test / ANOVA
            t_value = st.number_input("T-Value / F-Value", value=2.54)
            p_value = st.number_input("P-Value", value=0.01, format="%.3f")

with st.expander("B. RESEARCH DATA INPUT (EXCEL)"):
    uploaded_file = st.file_uploader("Unggah file Excel hasil olah data", type=["xlsx"])
    doi_list = st.text_area("C. Upload DOI Pendukung (Pisahkan dengan koma)")

# --- 5. VISUALIZER FUNCTION ---
def st_mermaid(code):
    st.components.v1.html(
        f"""
        <div class="mermaid" style="display: flex; justify-content: center;">{code}</div>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});
        </script>
        """, height=350,
    )

def render_model(tool, x, m, y, b1, b2, b3):
    if "SEM" in tool:
        code = f'graph LR\n  X["{x}"] -- "β={b1}" --> M["{m}"]\n  M -- "β={b2}" --> Y["{y}"]\n  X -. "β={b3}" .-> Y'
    elif "Regression" in tool:
        code = f'graph LR\n  X["{x}"] -- "β={b3}" --> Y["{y}"]'
    else: # T-Test
        code = f'graph TD\n  X["Kelompok {x}"] -- "Sig. Comparison" --> Y["Skor {y}"]'
    
    st_mermaid(code)

# --- 6. EKSEKUSI ---
if st.button("🚀 EXECUTE FULL RESEARCH SUITE"):
    if not iv or not dv:
        st.warning("Mohon lengkapi nama variabel.")
    else:
        tabs = st.tabs(["📊 Research Model", "📝 IMRAD Draft", "🔍 Deep Review", "📚 References"])

        with tabs[0]:
            st.subheader(f"Model: {tool}")
            render_model(tool, iv, mv, dv, beta_xm, beta_my, beta_xy)
            
            # Tabel ringkasan yang juga adaptif
            if "SEM" in tool:
                res_data = {"Path": [f"{iv}→{mv}", f"{mv}→{dv}", f"{iv}→{dv}"], "Coeff": [beta_xm, beta_my, beta_xy]}
            else:
                res_data = {"Variable": [iv], "Impact on": [dv], "Value": [beta_xy if "Regression" in tool else t_value]}
            st.table(pd.DataFrame(res_data))

        with tabs[1]:
            with st.spinner("Drafting IMRAD..."):
                prompt = f"Write a Scopus Q1 draft. Tool: {tool}. Vars: X={iv}, M={mv}, Y={dv}. Values: {beta_xm}, {beta_my}, {beta_xy}. Standard: Elsevier."
                res = safe_generate(model_instance, prompt)
                if res: st.code(res.text, language="markdown")

        with tabs[2]:
            st.info("Fitur Deep Review (Analisis DOI) sedang aktif...")
            # Masukkan prompt ekstraksi tabel CSV # Bapak di sini

        with tabs[3]:
            res = safe_generate(model_instance, f"Format DOI to APA 7: {doi_list}")
            if res: st.code(res.text, language="text")

st.divider()
st.caption("Adaptive Research Engine | AI Edition")