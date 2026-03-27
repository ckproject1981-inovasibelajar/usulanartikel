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
    st.error("❌ API Key missing!")
    st.stop()

# --- 4. UI INPUT ---
st.title("🎓 Education AI: Scopus Q1 Full Suite + Visualizer")

with st.expander("A. SETTING VARIABEL & STATISTIK", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        iv = st.text_input("Independent Variable (X)", value="Digital Literacy")
        mv = st.text_input("Mediator (M)", value="Teacher Self-Efficacy")
        dv = st.text_input("Dependent Variable (Y)", value="Student Engagement")
    with col2:
        tool = st.selectbox("Alat Statistik", ["PLS-SEM (SmartPLS)", "CB-SEM (AMOS)", "Multiple Regression"])
        # Tambahkan input manual untuk koefisien jika data excel belum diunggah
        st.write("**Manual Path Coefficients (Optional)**")
        c1, c2, c3 = st.columns(3)
        beta_xm = c1.number_input("β (X→M)", value=0.0)
        beta_my = c2.number_input("β (M→Y)", value=0.0)
        beta_xy = c3.number_input("β (X→Y)", value=0.0)

with st.expander("B. DATA SOURCE & REFERENCES"):
    uploaded_file = st.file_uploader("Unggah Hasil (.xlsx)", type=["xlsx"])
    doi_list = st.text_area("Input DOI (Pisahkan dengan koma)")

# --- 5. FUNGSI VISUALISASI (MERMAID) ---
def render_research_model(x, m, y, b1, b2, b3):
    # Logika arah panah dan label koefisien
    mermaid_code = f"""
    graph LR
        X["{x}"]
        M["{m}"]
        Y["{y}"]
        
        X -- "β={b1}" --> M
        M -- "β={b2}" --> Y
        X -. "Direct β={b3}" .-> Y
        
        style X fill:#f9f,stroke:#333,stroke-width:2px
        style M fill:#bbf,stroke:#333,stroke-width:2px
        style Y fill:#bfb,stroke:#333,stroke-width:2px
    """
    st.mermaid(mermaid_code)

# Custom fungsi untuk merender Mermaid di Streamlit
if not hasattr(st, 'mermaid'):
    def st_mermaid(code):
        st.components.v1.html(
            f"""
            <pre class="mermaid">
                {code}
            </pre>
            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{ startOnLoad: true }});
            </script>
            """,
            height=400,
        )
    st.mermaid = st_mermaid

# --- 6. EKSEKUSI ---
if st.button("🚀 EXECUTE FULL RESEARCH SUITE"):
    if not iv or not dv:
        st.warning("Mohon isi variabel minimal X dan Y.")
    else:
        tabs = st.tabs(["📊 Model & Path Visual", "📝 IMRAD Draft", "🔍 Deep Review", "📚 References"])

        with tabs[0]:
            st.subheader("Structural Model Visualization")
            # Jika ada file excel, AI bisa mencoba mengekstrak Beta otomatis (Fitur lanjutan)
            render_research_model(iv, mv, dv, beta_xm, beta_my, beta_xy)
            
            st.info("💡 Garis putus-putus menunjukkan efek langsung (Direct Effect), garis tegas menunjukkan jalur mediasi.")
            
            # Tambahkan tabel deskripsi Path
            df_path = pd.DataFrame({
                "Relationship": [f"{iv} -> {mv}", f"{mv} -> {dv}", f"{iv} -> {dv} (Direct)"],
                "Coefficient (β)": [beta_xm, beta_my, beta_xy],
                "Type": ["Mediation Path", "Mediation Path", "Direct Effect"]
            })
            st.table(df_path)

        with tabs[1]:
            with st.spinner("Drafting IMRAD..."):
                res = safe_generate(model_instance, f"Write Scopus Q1 article draft. IV:{iv}, MV:{mv}, DV:{dv}. Tool:{tool}. Standard: Elsevier.")
                if res: st.code(res.text, language="markdown")

        with tabs[2]:
            # Integrasi instruksi Deep Review Bapak yang sebelumnya
            st.write("Proses ekstraksi rujukan teori berdasarkan DOI...")
            # (Prompt ekstraksi tabel CSV # Bapak bisa dimasukkan di sini)

        with tabs[3]:
            res = safe_generate(model_instance, f"Format DOI to APA 7: {doi_list}")
            if res: st.code(res.text, language="text")

st.divider()
st.caption("Visual Research Engine | Path Analysis Viewer")