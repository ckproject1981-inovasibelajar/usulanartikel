import streamlit as st
import pandas as pd
import numpy as np
import io
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from semopy import Model, calc_stats
from scipy import stats

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="SEM Pro Assistant Super Ultimate", layout="wide", page_icon="🔬")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #d1d1d1; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .report-box { background-color: #ffffff; padding: 20px; border-radius: 10px; border-left: 5px solid #007bff; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    ai_model = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. HIGH-CONVERGENCE DATA GENERATOR ---
def generate_high_fit_data(n_x, n_m, n_y):
    rows = 450  # Sampel lebih besar untuk stabilitas
    data = {}
    
    # Generate Latent dengan Korelasi Kuat
    L_X_combined = np.random.normal(3.5, 0.6, rows)
    latents_x = {f"X{i}": 0.8 * L_X_combined + np.random.normal(0, 0.3, rows) for i in range(1, n_x + 1)}
    
    sum_x = sum(latents_x.values()) / n_x
    latents_m = {f"M{i}": 0.75 * sum_x + np.random.normal(0, 0.2, rows) for i in range(1, n_m + 1)}
    
    base_for_y = sum(latents_m.values())/n_m if n_m > 0 else sum_x
    latents_y = {f"Y{i}": 0.65 * base_for_y + 0.25 * sum_x + np.random.normal(0, 0.2, rows) for i in range(1, n_y + 1)}

    # Fungsi penambahan indikator dengan Loading Factor tinggi (~0.85)
    def add_inds(val, prefix, count):
        for i in range(1, 4):
            data[f"{prefix}{count}_{i}"] = np.clip(0.85 * val + np.random.normal(0, 0.15, rows), 1, 5)

    for i in range(1, n_x + 1): add_inds(latents_x[f"X{i}"], "X", i)
    for i in range(1, n_m + 1): add_inds(latents_m[f"M{i}"], "M", i)
    for i in range(1, n_y + 1): add_inds(latents_y[f"Y{i}"], "Y", i)
    
    df = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# --- 3. ANALYTICS TOOLS ---
def get_ave_cr_table(inspected, latent_dict):
    results = []
    loadings = inspected[inspected['op'] == '~=']
    for latent in latent_dict.keys():
        l_vals = loadings[loadings['lval'] == latent]['Estimate'].values
        if len(l_vals) > 0:
            ave = np.mean(np.square(l_vals))
            cr = np.sum(l_vals)**2 / (np.sum(l_vals)**2 + np.sum(1 - np.square(l_vals)))
            results.append({"Konstruk": latent, "AVE (>0.5)": round(ave, 3), "CR (>0.7)": round(cr, 3)})
    return pd.DataFrame(results)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("🔬 SEM Engine v3.0")
    st.markdown("---")
    
    with st.expander("🌟 Generator Data (Pasti Fit)"):
        nx = st.number_input("Variabel X", 1, 10, 3)
        nm = st.number_input("Variabel M", 0, 10, 1)
        ny = st.number_input("Variabel Y", 1, 10, 2)
        st.download_button("📥 Download Template High-Fit", 
                          generate_high_fit_data(nx, nm, ny), 
                          "template_high_fit.xlsx")

    uploaded_file = st.file_uploader("Upload Data Riset (.xlsx)", type=["xlsx"])
    st.caption("Dikembangkan oleh Citra Kurniawan & PUI DLI - 2026")

# --- 5. MAIN CONTENT ---
if uploaded_file:
    # Pre-processing Agresif
    df_raw = pd.read_excel(uploaded_file).ffill().bfill()
    df = df_raw.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    df = df.loc[:, (df.std() > 0.01)] # Buat kolom yang hampir konstan
    
    all_cols = sorted(list(set([c.split('_')[0] for c in df.columns if '_' in c])))
    
    st.header("📐 Pemodelan Struktural")
    c1, c2, c3 = st.columns(3)
    with c1: vx = st.multiselect("Exogenous (X)", all_cols)
    with c2: vm = st.multiselect("Mediators (M)", all_cols)
    with c3: vy = st.multiselect("Endogenous (Y)", all_cols)

    if vx and vy:
        # Generate Syntax
        m_syntax = "# Measurement\n"
        latent_map = {}
        for v in (vx + vm + vy):
            inds = [c for c in df.columns if c.startswith(v + "_")]
            m_syntax += f"{v} =~ {' + '.join(inds)}\n"
            latent_map[v] = inds
        
        s_syntax = "# Structural\n"
        for m in vm:
            for x in vx: s_syntax += f"{m} ~ {x}\n"
        for y in vy:
            for m in (vx + vm): s_syntax += f"{y} ~ {m}\n"
        
        if st.button("🏁 Jalankan SEM Analysis"):
            with st.spinner("Mengoptimasi Matriks Kovarians..."):
                try:
                    # Core SEM
                    model = Model(m_syntax + s_syntax)
                    model.fit(df)
                    inspected = model.inspect()
                    
                    # Safe Stats Retrieval
                    try:
                        stats_res = calc_stats(model).T
                    except:
                        stats_res = pd.DataFrame()

                    st.divider()
                    
                    # --- METRICS SECTION ---
                    if not stats_res.empty and 0 in stats_res.columns:
                        st.subheader("📊 Goodness of Fit Index")
                        m1, m2, m3, m4 = st.columns(4)
                        
                        def get_idx(key): return stats_res.loc[key, 0] if key in stats_res.index else 0
                        
                        cfi, rmsea, srmr, chi, dof = get_idx('CFI'), get_idx('RMSEA'), get_idx('SRMR'), get_idx('Chi-square'), get_idx('doF')
                        
                        m1.metric("CFI", f"{cfi:.3f}", "Good" if cfi > 0.9 else "Poor")
                        m2.metric("RMSEA", f"{rmsea:.3f}", "Good" if rmsea < 0.08 else "Poor")
                        m3.metric("SRMR", f"{srmr:.3f}", "Good" if srmr < 0.08 else "Poor")
                        m4.metric("CMIN/DF", f"{chi/dof:.2f}" if dof > 0 else "N/A")

                        # --- TABBED RESULTS ---
                        t1, t2, t3, t4, t5 = st.tabs(["🖼️ Diagram Jalur", "🔍 Validitas/CFA", "📈 Distribusi", "📋 Tabel Koefisien", "🤖 AI Report"])
                        
                        with t1:
                            dot = graphviz.Digraph(graph_attr={'rankdir':'LR', 'splines':'true'})
                            for v in (vx + vm + vy):
                                fill = '#D1E8FF' if v in vx else ('#D1FFD7' if v in vm else '#FFE8D1')
                                dot.node(v, v, shape='ellipse', style='filled', color='#333333', fillcolor=fill)
                            
                            paths = inspected[inspected['op'] == '~']
                            for _, r in paths.iterrows():
                                star = "***" if r['p-val'] < 0.001 else ("**" if r['p-val'] < 0.01 else ("*" if r['p-val'] < 0.05 else ""))
                                dot.edge(r['rval'], r['lval'], label=f"{r['Estimate']:.2f}{star}")
                            st.graphviz_chart(dot)

                        with t2:
                            st.write("### Analisis CFA & Reliability")
                            st.dataframe(get_ave_cr_table(inspected, latent_map), use_container_width=True)
                            st.info("Standar: AVE > 0.5 & CR > 0.7")

                        with t3:
                            st.write("### Cek Normalitas Indikator")
                            sel_ind = st.selectbox("Pilih Indikator:", df.columns)
                            fig, ax = plt.subplots(figsize=(10, 4))
                            sns.histplot(df[sel_ind], kde=True, color="#007bff", ax=ax)
                            st.pyplot(fig)

                        with t4:
                            st.write("### Regression Weights (Direct Effects)")
                            st.dataframe(paths.style.background_gradient(subset=['Estimate'], cmap='Blues'))

                        with t5:
                            if st.button("Generate Academic Draft"):
                                if "ai_model" in locals():
                                    sig_rel = paths[paths['p-val'] < 0.05]
                                    txt_path = ", ".join([f"{r['rval']}→{r['lval']}" for _, r in sig_rel.iterrows()])
                                    prompt = f"Interpretasikan hasil SEM ini: CFI={cfi:.3f}, RMSEA={rmsea:.3f}. Hubungan signifikan: {txt_path}. Tulis dalam gaya jurnal ilmiah."
                                    st.markdown(f"<div class='report-box'>{ai_model.generate_content(prompt).text}</div>", unsafe_allow_html=True)
                                else:
                                    st.warning("API Key tidak ditemukan.")
                    else:
                        st.error("❌ Model Gagal Konvergen.")
                        st.warning("Data dummy Anda terlalu acak atau variabel terlalu banyak. Gunakan tombol 'Download Template' di sidebar untuk mendapatkan data yang sudah tersinkronisasi sempurna.")

                except Exception as e:
                    st.error(f"🚨 System Error: {e}")
else:
    st.info("💡 Gunakan generator di sidebar untuk membuat data template, lalu unggah kembali untuk melihat analisis lengkap.")