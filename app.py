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

# --- 2. HIGH-STABILITY DATA GENERATOR ---
def generate_stable_data(n_x, n_m, n_y):
    # Menggunakan jumlah baris lebih banyak untuk menjamin konvergensi
    rows = 500 
    data = {}
    
    # Membuat 'True Latent Score' yang saling berhubungan kuat
    # Laten X (Independen)
    latents_x = {f"X{i}": np.random.normal(3.5, 0.5, rows) for i in range(1, n_x + 1)}
    
    # Laten M (Mediator) - sangat dipengaruhi oleh rata-rata X
    avg_x = sum(latents_x.values()) / n_x
    latents_m = {f"M{i}": 0.8 * avg_x + np.random.normal(0, 0.2, rows) for i in range(1, n_m + 1)}
    
    # Laten Y (Dependen) - dipengaruhi M dan X
    avg_m = sum(latents_m.values()) / n_m if n_m > 0 else avg_x
    latents_y = {f"Y{i}": 0.7 * avg_m + 0.2 * avg_x + np.random.normal(0, 0.2, rows) for i in range(1, n_y + 1)}

    # Fungsi untuk membuat indikator dengan Loading Factor tinggi (> 0.8)
    def add_indicators(latent_val, prefix, count):
        for i in range(1, 4): # 3 indikator per laten
            # Rumus: Y = 0.9*X + error (Sangat bersih untuk SEM)
            col_name = f"{prefix}{count}_{i}"
            measure = 0.9 * latent_val + np.random.normal(0, 0.1, rows)
            data[col_name] = np.clip(measure, 1, 5)

    # Isi data dengan indikator
    for i in range(1, n_x + 1): add_indicators(latents_x[f"X{i}"], "X", i)
    for i in range(1, n_m + 1): add_indicators(latents_m[f"M{i}"], "M", i)
    for i in range(1, n_y + 1): add_indicators(latents_y[f"Y{i}"], "Y", i)
    
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
    st.title("🔬 SEM Engine v3.1")
    st.markdown("---")
    
    with st.expander("🌟 Generator Data (Sangat Stabil)"):
        nx_val = st.number_input("Variabel X", 1, 10, 3)
        nm_val = st.number_input("Variabel M", 0, 10, 1)
        ny_val = st.number_input("Variabel Y", 1, 10, 2)
        st.download_button("📥 Download Template SEM", 
                          generate_stable_data(nx_val, nm_val, ny_val), 
                          "template_stabil.xlsx")

    uploaded_file = st.file_uploader("Upload Data Riset (.xlsx)", type=["xlsx"])
    st.caption("Dikembangkan oleh Citra Kurniawan & PUI DLI - 2026")

# --- 5. MAIN CONTENT ---
if uploaded_file:
    df_raw = pd.read_excel(uploaded_file).ffill().bfill()
    # Pembersihan kolom teks dan kolom dengan varians nol
    df = df_raw.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    df = df.loc[:, (df.std() > 0.05)] 
    
    all_cols = sorted(list(set([c.split('_')[0] for c in df.columns if '_' in c])))
    
    st.header("📐 Pemodelan Struktural")
    c1, c2, c3 = st.columns(3)
    with c1: vx = st.multiselect("Exogenous (X)", all_cols, help="Variabel yang mempengaruhi")
    with c2: vm = st.multiselect("Mediators (M)", all_cols, help="Variabel perantara")
    with c3: vy = st.multiselect("Endogenous (Y)", all_cols, help="Variabel yang dipengaruhi")

    if vx and vy:
        # Pembangun Syntax Model
        m_syntax = "# Measurement\n"
        latent_map = {}
        for v in (vx + vm + vy):
            inds = [c for c in df.columns if c.startswith(v + "_")]
            if inds:
                m_syntax += f"{v} =~ {' + '.join(inds)}\n"
                latent_map[v] = inds
        
        s_syntax = "# Structural\n"
        for m in vm:
            for x in vx: s_syntax += f"{m} ~ {x}\n"
        for y in vy:
            for m in vm: s_syntax += f"{y} ~ {m}\n"
            for x in vx: s_syntax += f"{y} ~ {x}\n"
        
        if st.button("🏁 Jalankan SEM Analysis"):
            with st.spinner("Sedang menghitung model..."):
                try:
                    model = Model(m_syntax + s_syntax)
                    model.fit(df)
                    inspected = model.inspect()
                    
                    try:
                        stats_res = calc_stats(model).T
                    except:
                        stats_res = pd.DataFrame()

                    if not stats_res.empty and 0 in stats_res.columns:
                        st.divider()
                        st.subheader("📊 Goodness of Fit Index")
                        m1, m2, m3, m4 = st.columns(4)
                        
                        def get_idx(key): return stats_res.loc[key, 0] if key in stats_res.index else 0
                        cfi, rmsea, srmr, chi, dof = get_idx('CFI'), get_idx('RMSEA'), get_idx('SRMR'), get_idx('Chi-square'), get_idx('doF')
                        
                        m1.metric("CFI", f"{cfi:.3f}", "Lulus" if cfi > 0.9 else "Gagal")
                        m2.metric("RMSEA", f"{rmsea:.3f}", "Lulus" if rmsea < 0.08 else "Gagal")
                        m3.metric("SRMR", f"{srmr:.3f}", "Lulus" if srmr < 0.08 else "Gagal")
                        m4.metric("CMIN/DF", f"{chi/dof:.2f}" if dof > 0 else "N/A")

                        tabs = st.tabs(["🖼️ Path Diagram", "🔍 Validitas", "📈 Distribusi", "📋 Koefisien Jalur", "🤖 AI Interpretasi"])
                        
                        with tabs[0]:
                            dot = graphviz.Digraph(graph_attr={'rankdir':'LR'})
                            for v in (vx + vm + vy):
                                fill = '#D1E8FF' if v in vx else ('#D1FFD7' if v in vm else '#FFE8D1')
                                dot.node(v, v, shape='ellipse', style='filled', fillcolor=fill)
                            
                            paths = inspected[inspected['op'] == '~']
                            for _, r in paths.iterrows():
                                star = "***" if r['p-val'] < 0.001 else ("**" if r['p-val'] < 0.01 else ("*" if r['p-val'] < 0.05 else ""))
                                dot.edge(r['rval'], r['lval'], label=f"{r['Estimate']:.2f}{star}")
                            st.graphviz_chart(dot)

                        with tabs[1]:
                            st.write("### Konstruk AVE & CR")
                            st.table(get_ave_cr_table(inspected, latent_map))

                        with tabs[2]:
                            sel_ind = st.selectbox("Pilih Indikator:", df.columns)
                            fig, ax = plt.subplots(figsize=(8, 3))
                            sns.histplot(df[sel_ind], kde=True, color="#007bff", ax=ax)
                            st.pyplot(fig)

                        with tabs[3]:
                            st.write("### Regression Weights")
                            st.dataframe(paths)

                        with tabs[4]:
                            if st.button("Generate Draft Narasi"):
                                if "ai_model" in locals():
                                    sig_rel = paths[paths['p-val'] < 0.05]
                                    txt_path = ", ".join([f"{r['rval']} ke {r['lval']}" for _, r in sig_rel.iterrows()])
                                    prompt = f"Analisis hasil SEM ini secara akademis: CFI {cfi:.3f}, Hubungan signifikan: {txt_path}. Tulis dalam Bahasa Indonesia formal."
                                    st.markdown(f"<div class='report-box'>{ai_model.generate_content(prompt).text}</div>", unsafe_allow_html=True)

                    else:
                        st.error("⚠️ Estimasi Gagal Konvergen. Pastikan data diunggah dengan format yang benar (Gunakan template dari sidebar).")
                except Exception as e:
                    st.error(f"🚨 Kesalahan: {e}")
else:
    st.info("💡 Selamat datang! Download template di sidebar, isi, lalu upload kembali untuk memulai analisis SEM.")