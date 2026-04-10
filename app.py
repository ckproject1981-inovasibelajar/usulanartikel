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
st.set_page_config(page_title="SEM Pro Assistant v3.2", layout="wide", page_icon="🔬")

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    ai_model = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. SUPER-STABLE DATA GENERATOR ---
def generate_bulletproof_data(n_x, n_m, n_y):
    rows = 500
    data = {}
    # Base Latent (Core correlation)
    base = np.random.normal(3, 0.5, rows)
    
    # Latents with extremely high correlation for guaranteed fit
    lx = {f"X{i}": base + np.random.normal(0, 0.1, rows) for i in range(1, n_x + 1)}
    lm = {f"M{i}": 0.8 * base + np.random.normal(0, 0.1, rows) for i in range(1, n_m + 1)}
    ly = {f"Y{i}": 0.7 * base + np.random.normal(0, 0.1, rows) for i in range(1, n_y + 1)}

    def add_inds(l_val, prefix, count):
        for i in range(1, 4):
            # 0.95 factor ensures almost zero chance of non-convergence
            data[f"{prefix}{count}_{i}"] = np.clip(0.95 * l_val + np.random.normal(0, 0.05, rows), 1, 5)

    for i in range(1, n_x + 1): add_inds(lx[f"X{i}"], "X", i)
    for i in range(1, n_m + 1): add_inds(lm[f"M{i}"], "M", i)
    for i in range(1, n_y + 1): add_inds(ly[f"Y{i}"], "Y", i)
    
    df = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🔬 SEM Engine v3.2")
    with st.expander("🌟 Generator Template"):
        nx_v = st.number_input("Variabel X", 1, 5, 3)
        nm_v = st.number_input("Variabel M", 0, 5, 1)
        ny_v = st.number_input("Variabel Y", 1, 5, 2)
        st.download_button("📥 Download Template", generate_bulletproof_data(nx_v, nm_v, ny_v), "data_sem.xlsx")
    uploaded_file = st.file_uploader("Upload Data (.xlsx)", type=["xlsx"])
    st.caption("PUI-PT DLI UM - 2026")

# --- 4. MAIN LOGIC ---
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    # CLEANING: Hapus spasi di nama kolom
    df.columns = [str(c).strip() for c in df.columns]
    df = df.apply(pd.to_numeric, errors='coerce').ffill().bfill()
    
    # Deteksi variabel yang tersedia (X, M, Y)
    available_vars = sorted(list(set([c.split('_')[0] for c in df.columns if '_' in c])))
    
    st.header("📐 Model Specification")
    c1, c2, c3 = st.columns(3)
    with c1: vx = st.multiselect("Exogenous (X)", available_vars)
    with c2: vm = st.multiselect("Mediators (M)", available_vars)
    with c3: vy = st.multiselect("Endogenous (Y)", available_vars)

    if vx and vy:
        # Build Syntax
        m_syntax = ""
        latent_map = {}
        for v in (vx + vm + vy):
            inds = [c for c in df.columns if c.startswith(v + "_")]
            if len(inds) >= 2: # Syarat minimal 2 indikator
                m_syntax += f"{v} =~ {' + '.join(inds)}\n"
                latent_map[v] = inds
        
        s_syntax = ""
        for m in vm:
            for x in vx: s_syntax += f"{m} ~ {x}\n"
        for y in vy:
            for m in (vx + vm): s_syntax += f"{y} ~ {m}\n"

        if st.button("🏁 Run Analysis"):
            try:
                model = Model(m_syntax + s_syntax)
                res_fit = model.fit(df)
                # Validasi jika semopy gagal internal
                if "Optimization terminated successfully" not in str(res_fit):
                    st.warning("⚠️ Optimasi tidak sempurna, hasil mungkin bias.")
                
                inspected = model.inspect()
                stats_res = calc_stats(model).T
                
                # --- HASIL ---
                st.divider()
                st.subheader("📊 Goodness of Fit Index")
                m1, m2, m3, m4 = st.columns(4)
                
                def get_s(k): return stats_res.loc[k, 0] if k in stats_res.index else 0
                cfi, rmsea, srmr, chi, dof = get_s('CFI'), get_s('RMSEA'), get_s('SRMR'), get_s('Chi-square'), get_s('doF')
                
                m1.metric("CFI", f"{cfi:.3f}")
                m2.metric("RMSEA", f"{rmsea:.3f}")
                m3.metric("SRMR", f"{srmr:.3f}")
                m4.metric("CMIN/DF", f"{chi/dof:.2f}" if dof > 0 else "N/A")

                # Tabs
                t1, t2, t3 = st.tabs(["🖼️ Diagram", "📋 Koefisien", "🤖 AI Interpretasi"])
                with t1:
                    dot = graphviz.Digraph(graph_attr={'rankdir':'LR'})
                    for v in (vx + vm + vy):
                        dot.node(v, v, shape='ellipse', style='filled', fillcolor='#E3F2FD')
                    for _, r in inspected[inspected['op'] == '~'].iterrows():
                        dot.edge(r['rval'], r['lval'], label=f"{r['Estimate']:.2f}")
                    st.graphviz_chart(dot)
                with t2:
                    st.dataframe(inspected)
                with t3:
                    if "ai_model" in locals():
                        prompt = f"Interpretasikan hasil SEM: CFI {cfi:.3f}, RMSEA {rmsea:.3f}."
                        st.write(ai_model.generate_content(prompt).text)

            except Exception as e:
                st.error(f"Gagal Konvergen: {e}")
                st.info("💡 Pastikan Variabel yang Anda pilih di menu 'Model Specification' sesuai dengan data di file Excel.")
else:
    st.info("Download template di sidebar, lalu upload kembali untuk mencoba.")