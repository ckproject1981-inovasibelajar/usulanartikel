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

# --- 1. INITIALIZATION ---
st.set_page_config(page_title="SEM Pro Assistant Ultimate", layout="wide", page_icon="🔬")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

# AI Configuration
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    ai_model = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. ADVANCED DUMMY GENERATOR ---
def generate_synced_data(n_x=3, n_m=3, n_y=3):
    rows = 300
    data = {}
    L_X = np.random.normal(3.5, 0.7, rows)
    L_M = 0.6 * L_X + np.random.normal(0, 0.4, rows)
    L_Y = 0.5 * L_M + 0.3 * L_X + np.random.normal(0, 0.4, rows)
    
    def add_indicators(latent, prefix, count):
        for i in range(1, 4):
            col_name = f"{prefix}{count}_{i}"
            data[col_name] = np.clip(0.8 * latent + np.random.normal(0, 0.3, rows), 1, 5)

    for i in range(1, n_x + 1): add_indicators(L_X, "X", i)
    for i in range(1, n_m + 1): add_indicators(L_M, "M", i)
    for i in range(1, n_y + 1): add_indicators(L_Y, "Y", i)
    
    df = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

def get_ave_cr(inspected, latent_dict):
    results = []
    loadings = inspected[inspected['op'] == '~=']
    for latent, inds in latent_dict.items():
        l_vals = loadings[loadings['lval'] == latent]['Estimate'].values
        if len(l_vals) > 0:
            ave = np.mean(np.square(l_vals))
            cr = np.sum(l_vals)**2 / (np.sum(l_vals)**2 + np.sum(1 - np.square(l_vals)))
            results.append({"Construct": latent, "AVE": round(ave, 3), "CR": round(cr, 3)})
    return pd.DataFrame(results)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🔬 SEM Engine v2.1")
    with st.expander("🛠️ Dummy Data Settings"):
        nx_in = st.number_input("Jumlah X", 1, 5, 3)
        nm_in = st.number_input("Jumlah M", 1, 5, 3)
        ny_in = st.number_input("Jumlah Y", 1, 5, 3)
        st.download_button("📥 Download Template", generate_synced_data(nx_in, nm_in, ny_in), "research_data.xlsx")
    uploaded_file = st.file_uploader("Upload Research Data", type=["xlsx"])

# --- 4. MAIN LOGIC ---
if uploaded_file:
    # Load and Pre-process Data
    df_raw = pd.read_excel(uploaded_file).ffill().bfill()
    
    # PERBAIKAN: Konversi semua kolom ke numerik dan buang yang tidak bisa dikonversi
    df = df_raw.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    
    # PERBAIKAN: Buang kolom dengan varians nol (konstan) agar tidak menyebabkan error "division by zero" di SEM
    df = df.loc[:, (df != df.iloc[0]).any()]
    
    if df.empty:
        st.error("❌ Data kosong setelah pembersihan. Pastikan kolom indikator berisi angka.")
        st.stop()

    cols = sorted(list(set([c.split('_')[0] for c in df.columns if '_' in c])))
    
    st.header("📐 Model Specification")
    c1, c2, c3 = st.columns(3)
    with c1: vx = st.multiselect("Exogenous (X)", cols)
    with c2: vm = st.multiselect("Mediators (M)", cols)
    with c3: vy = st.multiselect("Endogenous (Y)", cols)

    if vx and vy:
        # Build Syntax
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
            for m in (vx + vm): s_syntax += f"{y} ~ {m}\n"
        
        if st.button("🏁 Run Analysis"):
            try:
                # Execution
                model = Model(m_syntax + s_syntax)
                model.fit(df)
                inspected = model.inspect()
                stats_res = calc_stats(model).T
                
                # --- RESULTS ---
                st.divider()
                st.subheader("📊 Model Diagnostics & Goodness of Fit")
                m1, m2, m3, m4 = st.columns(4)
                
                # Validasi nilai indeks
                cfi = stats_res.loc['CFI', 0] if 'CFI' in stats_res.index else 0
                rmsea = stats_res.loc['RMSEA', 0] if 'RMSEA' in stats_res.index else 0
                srmr = stats_res.loc['SRMR', 0] if 'SRMR' in stats_res.index else 0
                cmin_df = (stats_res.loc['Chi-square', 0] / stats_res.loc['doF', 0]) if 'doF' in stats_res.index and stats_res.loc['doF', 0] != 0 else 0
                
                m1.metric("CFI (Ref > 0.90)", f"{cfi:.3f}", "✅" if cfi >= 0.9 else "⚠️")
                m2.metric("RMSEA (Ref < 0.08)", f"{rmsea:.3f}", "✅" if rmsea <= 0.08 else "⚠️")
                m3.metric("SRMR (Ref < 0.08)", f"{srmr:.3f}", "✅" if srmr <= 0.08 else "⚠️")
                m4.metric("CMIN/DF (Ref < 3.0)", f"{cmin_df:.2f}", "✅" if cmin_df <= 3 else "⚠️")

                tabs = st.tabs(["📉 Structural Model", "🔍 Measurement (CFA)", "📊 Data Distribution", "💎 Quality", "📝 AI Writer"])

                with tabs[0]:
                    st.write("### Full Path Diagram")
                    dot = graphviz.Digraph()
                    dot.attr(rankdir='LR')
                    for v in (vx + vm + vy):
                        color = '#E3F2FD' if v in vx else ('#E8F5E9' if v in vm else '#FFF3E0')
                        dot.node(v, v, shape='ellipse', style='filled', fillcolor=color)
                    paths = inspected[inspected['op'] == '~']
                    for _, r in paths.iterrows():
                        label = f"{r['Estimate']:.2f}" + ("*" if r['p-val'] < 0.05 else "")
                        dot.edge(r['rval'], r['lval'], label=label)
                    st.graphviz_chart(dot)

                with tabs[1]:
                    st.write("### CFA Measurement Model")
                    sel_v = st.selectbox("Pilih Variabel Laten:", vx + vm + vy)
                    cfa_dot = graphviz.Digraph()
                    cfa_dot.node(sel_v, sel_v, shape='ellipse', style='filled', fillcolor='#D1C4E9')
                    loadings = inspected[(inspected['op'] == '~=') & (inspected['lval'] == sel_v)]
                    for _, r in loadings.iterrows():
                        cfa_dot.node(r['rval'], r['rval'], shape='box')
                        cfa_dot.edge(sel_v, r['rval'], label=f"λ={r['Estimate']:.2f}")
                    st.graphviz_chart(cfa_dot)

                with tabs[2]:
                    st.write("### Histogram & Normality Check")
                    target_col = st.selectbox("Pilih Indikator:", df.columns)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.histplot(df[target_col], kde=True, color="skyblue", ax=ax)
                    st.pyplot(fig)

                with tabs[3]:
                    st.table(get_ave_cr(inspected, latent_map))
                    st.write("### Path Coefficients")
                    st.dataframe(paths)

                with tabs[4]:
                    if "ai_model" in locals():
                        if st.button("Generate Draft Manuscript"):
                            with st.spinner("AI sedang meramu interpretasi..."):
                                prompt = f"Tuliskan interpretasi hasil SEM profesional. Fit: CFI {cfi:.3f}, RMSEA {rmsea:.3f}. Fokus pada hubungan signifikan p < 0.05."
                                st.write(ai_model.generate_content(prompt).text)

            except Exception as e:
                # PERBAIKAN: Detail error yang membantu proses debugging
                st.error(f"Detail Error: {type(e).__name__} - {e}")
                st.info("💡 Tip: Pastikan variabel yang dipilih memiliki minimal 3 indikator dan data tidak mengandung nilai yang sama semua.")
else:
    st.info("👋 Selamat Datang! Pilih jumlah variabel di sidebar, download template, lalu upload kembali untuk memulai.")