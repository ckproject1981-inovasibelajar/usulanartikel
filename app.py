import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import io
from docx import Document # Pustaka baru untuk ekspor Word
try:
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    import graphviz
except ImportError:
    st.error("⚠️ Pustaka sistem belum lengkap. Pastikan requirements.txt berisi: scipy, scikit-learn, graphviz, python-docx")

# --- 1. CONFIG & ENGINE ---
st.set_page_config(page_title="Q1 SEM Ultimate Pro", layout="wide", page_icon="🎓")

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
    st.error("❌ API Key Gemini tidak ditemukan.")
    st.stop()

# --- 2. CORE FUNCTIONS ---
def generate_multi_var_dummy():
    rows = 70
    data = {}
    for i in range(1, 4): # Disederhanakan 3 set agar file tidak terlalu berat
        base = np.random.randint(2, 5, rows)
        for j in range(1, 4): 
            data[f'X{i}_{j}'] = np.clip(base + np.random.normal(0, 0.5, rows), 1, 5).round()
            data[f'M{i}_{j}'] = np.clip(base * 0.5 + np.random.normal(0, 0.7, rows), 1, 5).round()
            data[f'Y{i}_{j}'] = np.clip(base * 0.4 + np.random.normal(0, 0.8, rows), 1, 5).round()
    df_dummy = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_dummy.to_excel(writer, index=False)
    return output.getvalue()

def create_word_report(title, content):
    doc = Document()
    doc.add_heading(title, 0)
    doc.add_paragraph(content)
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

# --- 3. UI LAYOUT ---
st.image("https://i.ibb.co.com/23N3kpBY/Logo-DLI.png", width=160)
st.title("🎓 Q1 SEM Ultimate: Full Analysis & Word Export")

with st.sidebar:
    st.image("https://i.ibb.co.com/23N3kpBY/Logo-DLI.png", use_container_width=True)
    st.header("📂 Data Center")
    st.download_button("📥 Download Excel Template", generate_multi_var_dummy(), "template_sem.xlsx")
    st.divider()
    uploaded_file = st.file_uploader("Unggah Data Riset (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file).ffill().bfill()
    st.subheader("1. Setup Variabel")
    c1, c2, c3 = st.columns(3)
    ax = [c1.text_input(f"X{i}", "X1" if i==1 else "") for i in range(1, 4)]
    am = [c2.text_input(f"M{i}", "M1" if i==1 else "") for i in range(1, 4)]
    ay = [c3.text_input(f"Y{i}", "Y1" if i==1 else "") for i in range(1, 4)]
    
    active_x = [x for x in ax if x]
    active_m = [m for m in am if m]
    active_y = [y for y in ay if y]
    
    if active_x and active_y:
        # Perhitungan Skor Rata-rata
        avg_scores = pd.DataFrame()
        for v in active_x + active_m + active_y:
            cols = [c for c in df.columns if c.startswith(v)]
            if cols: avg_scores[v] = df[cols].mean(axis=1)

        st.subheader("2. Visualisasi Structural Model")
        dot = graphviz.Digraph(engine='dot')
        dot.attr(rankdir='LR')
        
        # Gambar Jalur
        for y_v in active_y:
            preds = active_x + active_m
            reg = LinearRegression().fit(avg_scores[preds], avg_scores[y_v])
            coefs = dict(zip(preds, reg.coef_))
            for m_v in active_m: dot.edge(m_v, y_v, label=f"b={round(coefs[m_v],2)}", color='green')
            for x_v in active_x: dot.edge(x_v, y_v, label=f"c'={round(coefs[x_v],2)}", style='dashed', color='orange')
        
        for x_v in active_x:
            for m_v in active_m:
                slope, _, _, _, _ = stats.linregress(avg_scores[x_v], avg_scores[m_v])
                dot.edge(x_v, m_v, label=f"a={round(slope,2)}", color='blue')

        st.graphviz_chart(dot)

        if st.button("🚀 GENERATE MANUSCRIPT & DOCX"):
            with st.spinner("AI sedang menyusun artikel..."):
                summary_stats = avg_scores.corr().to_string()
                # PERBAIKAN SYNTAX DI SINI (TANDA KUTIP DITUTUP)
                prompt = f"Berikan interpretasi profesional hasil SEM: X:{active_x}, M:{active_m}, Y:{active_y}. Gunakan korelasi: {summary_stats}. Fokus pada efek langsung vs tidak langsung. Bahasa Indonesia."
                
                try:
                    result_text = model.generate_content(prompt).text
                    st.markdown(result_text)
                    
                    # Fitur Download Word
                    word_file = create_word_report("Laporan Analisis SEM Q1", result_text)
                    st.download_button("📝 Download Hasil (Word)", word_file, "Hasil_Analisis_SEM.docx")
                except Exception as e:
                    st.error(f"Error AI: {e}")
    else:
        st.error("Mohon isi minimal satu variabel X dan satu Y.")
else:
    st.warning("Silakan unggah file untuk memulai.")

st.divider()
st.caption("Finalized Suite Ver 4.3 | Developed by Citra Kurniawan - 2026")