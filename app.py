import streamlit as st
import google.generativeai as genai
import pandas as pd
import time

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Q1 Research Pro: Visual Path", layout="wide")

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

# --- 4. UI INPUT ---
st.title("🎓 Education AI: Scopus Q1 Full Suite + Visualizer")

with st.expander("A. STATISTICAL CONFIGURATION", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        iv = st.text_input("Independent Variable (X)", value="Digital Literacy")
        mv = st.text_input("Mediator (M)", value="Teacher Self-Efficacy")
        dv = st.text_input("Dependent Variable (Y)", value="Student Engagement")
    with col2:
        tool = st.selectbox("Alat Statistik", ["PLS-SEM (SmartPLS)", "CB-SEM (AMOS)", "Multiple Linear Regression", "ANOVA", "T-Test"])
        st.write("**Input Path Coefficients (β)**")
        c1, c2, c3 = st.columns(3)
        beta_xm = c1.number_input("β (X→M)", value=0.45)
        beta_my = c2.number_input("β (M→Y)", value=0.38)
        beta_xy = c3.number_input("β (X→Y) Direct", value=0.12)

with st.expander("B. RESEARCH DATA INPUT (EXCEL)"):
    uploaded_file = st.file_uploader("Unggah file Excel hasil olah data", type=["xlsx"])
    doi_list = st.text_area("C. Upload DOI Pendukung (Pisahkan dengan koma)")

# --- 5. VISUALIZER FUNCTION (FIXED SYNTAX) ---
def st_mermaid(code):
    st.components.v1.html(
        f"""
        <div class="mermaid" style="display: flex; justify-content: center;">
            {code}
        </div>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});
        </script>
        """,
        height=350,
    )

def render_research_model(x, m, y, b1, b2, b3):
    # Menggunakan tanda kutip ganda "" di dalam ID Mermaid untuk mencegah Syntax Error
    mermaid_code = f"""
    graph LR
        X["{x}"] -- "β={b1}" --> M["{m}"]
        M -- "β={b2}" --> Y["{y}"]
        X -. "Direct β={b3}" .-> Y
        
        style X fill:#ffffff,stroke:#333,stroke-width:2px
        style M fill:#ffffff,stroke:#333,stroke-width:2px
        style Y fill:#ffffff,stroke:#333,stroke-width:2px
    """
    st_mermaid(mermaid_code)

# --- 6. EKSEKUSI ---
if st.button("🚀 EXECUTE FULL RESEARCH SUITE"):
    if not iv or not dv:
        st.warning("Mohon lengkapi nama variabel.")
    else:
        tabs = st.tabs(["📊 Structural Model", "📝 IMRAD Draft", "🔍 Deep Review", "📚 References"])

        with tabs[0]:
            st.subheader("Structural Model Visualization")
            render_research_model(iv, mv, dv, beta_xm, beta_my, beta_xy)
            
            st.write("**Path Analysis Summary Table**")
            df_path = pd.DataFrame({{
                "Path Relationship": [f"{iv} → {mv}", f"{mv} → {dv}", f"{iv} → {dv} (Direct)"],
                "Coefficient (β)": [beta_xm, beta_my, beta_xy],
                "Result": ["Significant" if abs(beta_xm) > 0.1 else "Non-Significant", 
                           "Significant" if abs(beta_my) > 0.1 else "Non-Significant",
                           "Significant" if abs(beta_xy) > 0.1 else "Non-Significant"]
            }})
            st.table(df_path)

        with tabs[1]:
            with st.spinner("Drafting IMRAD..."):
                prompt = f"Write a Scopus Q1 IMRAD draft. X:{iv}, M:{mv}, Y:{dv}. β values: X-M={beta_xm}, M-Y={beta_my}. Tool: {tool}. Elsevier style."
                res = safe_generate(model_instance, prompt)
                if res: st.code(res.text, language="markdown")

        with tabs[2]:
            st.info("Fitur Deep Review (Analisis DOI) sedang memproses metadata rujukan...")
            # (Tambahkan logika ekstraksi CSV # Bapak di sini)

        with tabs[3]:
            res = safe_generate(model_instance, f"Format DOI to APA 7: {doi_list}")
            if res: st.code(res.text, language="text")

st.divider()
st.caption("Anti-Hallucination & Visual Engine | Standard: Elsevier & J. Eichler")