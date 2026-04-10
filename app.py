import streamlit as st
import pandas as pd
import numpy as np
import io
import graphviz
from semopy import Model, calc_stats

# --- CONFIG ---
st.set_page_config(page_title="SEM Pro Assistant v3.4", layout="wide")

# --- ULTRA-STABLE GENERATOR ---
def generate_stable_data(nx, nm, ny):
    rows = 600 # Sampel lebih besar untuk stabilitas statistik
    data = {}
    base = np.random.normal(3.5, 0.4, rows)
    
    def add_vars(prefix, count, latent_base):
        for i in range(1, count + 1):
            l_val = latent_base + np.random.normal(0, 0.05, rows)
            for j in range(1, 4): 
                data[f"{prefix}{i}_{j}"] = np.clip(0.95 * l_val + np.random.normal(0, 0.05, rows), 1, 5)

    add_vars("X", nx, base)
    add_vars("M", nm, 0.85 * base)
    add_vars("Y", ny, 0.75 * base)
    
    df = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🔬 SEM Engine v3.4")
    with st.expander("🛠️ Control Panel"):
        nx_i = st.number_input("Variabel X", 1, 5, 3)
        nm_i = st.number_input("Variabel M", 0, 5, 1)
        ny_i = st.number_input("Variabel Y", 1, 5, 2)
        st.download_button("📥 Get Template", generate_stable_data(nx_i, nm_i, ny_i), "data_sem.xlsx")
    uploaded = st.file_uploader("Upload Data", type=["xlsx"])

# --- MAIN ---
if uploaded:
    df = pd.read_excel(uploaded)
    df.columns = [str(c).strip().replace(' ', '') for c in df.columns]
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    
    available = sorted(list(set([c.split('_')[0] for c in df.columns if '_' in c])))
    
    st.header("📐 Model Setup")
    c1, c2, c3 = st.columns(3)
    with c1: vx = st.multiselect("Exogenous (X)", available)
    with c2: vm = st.multiselect("Mediators (M)", available)
    with c3: vy = st.multiselect("Endogenous (Y)", available)

    if vx and vy:
        m_syntax = ""
        for v in (vx + vm + vy):
            inds = [c for c in df.columns if c.startswith(v + "_")]
            if inds: m_syntax += f"{v} =~ {' + '.join(inds)}\n"
        
        s_syntax = ""
        for m in vm:
            for x in vx: s_syntax += f"{m} ~ {x}\n"
        for y in vy:
            for m in (vx + vm): s_syntax += f"{y} ~ {m}\n"

        if st.button("🏁 Run Final Analysis"):
            try:
                model = Model(m_syntax + s_syntax)
                model.fit(df)
                inspected = model.inspect()
                
                # --- SAFE STATS RETRIEVAL ---
                st.divider()
                try:
                    stats_df = calc_stats(model)
                    # Jika stats_df adalah Series, ubah ke DataFrame
                    if isinstance(stats_df, pd.Series):
                        stats_df = stats_df.to_frame().T
                    
                    st.subheader("📊 Goodness of Fit Index")
                    cols = st.columns(len(stats_df.columns[:4])) # Ambil 4 pertama
                    
                    for i, col_name in enumerate(stats_df.columns[:4]):
                        val = stats_df.iloc[0][col_name]
                        cols[i].metric(col_name, f"{val:.3f}")
                        
                    with st.expander("📈 Lihat Semua Metrik Fit"):
                        st.table(stats_df)

                except Exception as e_inner:
                    st.warning(f"Metrik Fit (CFI/RMSEA) tidak tersedia: {e_inner}")
                    st.info("Ini biasanya terjadi jika model terlalu sederhana (Just-Identified).")

                # --- PATH DIAGRAM ---
                st.subheader("🖼️ Path Diagram")
                dot = graphviz.Digraph(graph_attr={'rankdir':'LR'})
                for v in (vx + vm + vy):
                    dot.node(v, v, shape='ellipse', style='filled', fillcolor='#E3F2FD')
                
                paths = inspected[inspected['op'] == '~']
                for _, r in paths.iterrows():
                    label = f"{r['Estimate']:.2f}"
                    dot.edge(str(r['rval']), str(r['lval']), label=label)
                st.graphviz_chart(dot)

                st.subheader("📋 Tabel Koefisien Lengkap")
                st.dataframe(inspected)

            except Exception as e:
                st.error(f"❌ Kesalahan Fatal: {e}")