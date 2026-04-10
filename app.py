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

# --- 2. ADVANCED DUMMY GENERATOR (Synced for Fit Model) ---
def generate_synced_data(n_x=3, n_m=3, n_y=3):
    rows = 300
    data = {}
    # Base Latent
    L_X = np.random.normal(3.5, 0.7, rows)
    L_M = 0.6 * L_X + np.random.normal(0, 0.4, rows)
    L_Y = 0.5 * L_M + 0.3 * L_X + np.random.normal(0, 0.4, rows)
    
    def add_indicators(latent, prefix, count):
        for i in range(1, 4): # 3 indikator per laten
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
    st.title("🔬 SEM Engine v2.5")
    st.subheader("Pusat Pengaturan Data")
    
    with st.expander("🛠️ Generator Data Dummy"):
        nx_in = st.number_input("Jumlah Independen (X)", 1, 10, 3)
        nm_in = st.number_input("Jumlah Mediator (M)", 0, 10, 1)
        ny_in = st.number_input("Jumlah Dependen (Y)", 1, 10, 2)
        st.download_button("📥 Download Template Excel", 
                          generate_synced_data(nx_in, nm_in, ny_in), 
                          "template_sem_pro.xlsx")

    uploaded_file = st.file_uploader("Upload File Riset Anda (.xlsx)", type=["xlsx"])
    st.info("Pastikan format nama kolom: Variabel_Indikator (Contoh: X1_1, X1_2)")

# --- 4. MAIN LOGIC ---
if uploaded_file:
    # --- Data Cleaning ---
    df_raw = pd.read_excel(uploaded_file).ffill().bfill()
    # Pastikan numerik & buang kolom konstan (varians 0)
    df = df_raw.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    df = df.loc[:, (df != df.iloc[0]).any()]
    
    if df.empty:
        st.error("❌ Data tidak valid atau kosong setelah dibersihkan.")
        st.stop()

    cols = sorted(list(set([c.split('_')[0] for c in df.columns if '_' in c])))
    
    st.header("📐 Spesifikasi Model SEM")
    c1, c2, c3 = st.columns(3)
    with c1: vx = st.multiselect("Variabel Independen (X)", cols)
    with c2: vm = st.multiselect("Variabel Mediator (M)", cols)
    with c3: vy = st.multiselect("Variabel Dependen (Y)", cols)

    if vx and vy:
        # Build Syntax Dinamis
        m_syntax = "# Measurement Model\n"
        latent_map = {}
        for v in (vx + vm + vy):
            inds = [c for c in df.columns if c.startswith(v + "_")]
            if inds:
                m_syntax += f"{v} =~ {' + '.join(inds)}\n"
                latent_map[v] = inds
        
        s_syntax = "# Structural Model\n"
        for m in vm:
            for x in vx: s_syntax += f"{m} ~ {x}\n"
        for y in vy:
            for m in vm: s_syntax += f"{y} ~ {m}\n"
            for x in vx: s_syntax += f"{y} ~ {x}\n"
        
        full_syntax = m_syntax + s_syntax
        
        if st.button("🏁 Jalankan Analisis Sekarang"):
            with st.spinner("Sedang menghitung estimasi model..."):
                try:
                    # 1. Fit Model
                    model = Model(full_syntax)
                    model.fit(df)
                    inspected = model.inspect()
                    
                    # 2. Ambil Statistik dengan Proteksi KeyError
                    try:
                        stats_res = calc_stats(model).T
                    except:
                        stats_res = pd.DataFrame()

                    # --- RESULTS UI ---
                    st.divider()
                    st.subheader("📊 Diagnostik Model (Goodness of Fit)")
                    
                    if not stats_res.empty and 0 in stats_res.columns:
                        m1, m2, m3, m4 = st.columns(4)
                        def safe_stat(idx): return stats_res.loc[idx, 0] if idx in stats_res.index else None

                        cfi = safe_stat('CFI')
                        rmsea = safe_stat('RMSEA')
                        srmr = safe_stat('SRMR')
                        chi = safe_stat('Chi-square')
                        dof = safe_stat('doF')

                        m1.metric("CFI (Ref > 0.90)", f"{cfi:.3f}" if cfi is not None else "N/A", "✅" if cfi and cfi >= 0.9 else "⚠️")
                        m2.metric("RMSEA (Ref < 0.08)", f"{rmsea:.3f}" if rmsea is not None else "N/A", "✅" if rmsea and rmsea <= 0.08 else "⚠️")
                        m3.metric("SRMR (Ref < 0.08)", f"{srmr:.3f}" if srmr is not None else "N/A", "✅" if srmr and srmr <= 0.08 else "⚠️")
                        
                        if chi and dof and dof > 0:
                            m4.metric("CMIN/DF (Ref < 3.0)", f"{chi/dof:.2f}", "✅" if (chi/dof) <= 3 else "⚠️")
                        else:
                            m4.metric("CMIN/DF", "N/A")

                        # --- TABS ---
                        tabs = st.tabs(["📉 Diagram Jalur", "🔍 Analisis CFA", "📊 Sebaran Data", "💎 Validitas & Reliabilitas", "📝 Draft Laporan AI"])
                        
                        with tabs[0]:
                            st.write("### Structural Path Diagram")
                            dot = graphviz.Digraph()
                            dot.attr(rankdir='LR')
                            for v in (vx + vm + vy):
                                color = '#E3F2FD' if v in vx else ('#E8F5E9' if v in vm else '#FFF3E0')
                                dot.node(v, v, shape='ellipse', style='filled', fillcolor=color)
                            
                            paths = inspected[inspected['op'] == '~']
                            for _, r in paths.iterrows():
                                sig = "*" if r['p-val'] < 0.05 else ""
                                dot.edge(r['rval'], r['lval'], label=f"{r['Estimate']:.2f}{sig}")
                            st.graphviz_chart(dot)

                        with tabs[1]:
                            st.write("### Measurement Model Detail")
                            sel_v = st.selectbox("Pilih Konstruk Laten:", vx + vm + vy)
                            cfa_dot = graphviz.Digraph()
                            cfa_dot.node(sel_v, sel_v, shape='ellipse', style='filled', fillcolor='#D1C4E9')
                            loadings = inspected[(inspected['op'] == '~=') & (inspected['lval'] == sel_v)]
                            for _, r in loadings.iterrows():
                                cfa_dot.node(r['rval'], r['rval'], shape='box')
                                cfa_dot.edge(sel_v, r['rval'], label=f"λ={r['Estimate']:.2f}")
                            st.graphviz_chart(cfa_dot)

                        with tabs[2]:
                            st.write("### Uji Normalitas (Histogram)")
                            target_col = st.selectbox("Pilih Indikator untuk Dicek:", df.columns)
                            fig, ax = plt.subplots(figsize=(8, 4))
                            sns.histplot(df[target_col], kde=True, color="skyblue", ax=ax)
                            st.pyplot(fig)

                        with tabs[3]:
                            st.write("### Construct Reliability (AVE & CR)")
                            st.table(get_ave_cr(inspected, latent_map))
                            st.write("### Daftar Koefisien Jalur (Full Table)")
                            st.dataframe(inspected)

                        with tabs[4]:
                            if st.button("✍️ Buat Draft Artikel"):
                                if "ai_model" in locals():
                                    with st.spinner("AI sedang menganalisis hasil..."):
                                        p_sig = paths[paths['p-val'] < 0.05]
                                        p_txt = ", ".join([f"{r['rval']} ke {r['lval']}" for _, r in p_sig.iterrows()])
                                        prompt = f"Tuliskan laporan hasil penelitian SEM. Model Fit: CFI={cfi:.3f}. Hubungan signifikan pada: {p_txt}. Gunakan bahasa akademik Indonesia yang formal."
                                        st.write(ai_model.generate_content(prompt).text)
                                else:
                                    st.warning("API Key Gemini belum diatur di secrets.")
                    else:
                        st.error("⚠️ Estimasi Gagal Konvergen. Model terlalu kompleks atau data kurang variatif.")
                        st.info("Coba kurangi jumlah variabel atau gunakan data dummy untuk tes.")

                except Exception as e:
                    st.error(f"🚨 Kesalahan Teknis: {type(e).__name__} - {e}")

else:
    st.info("💡 Silakan unduh template di sidebar, isi dengan data Anda, lalu unggah kembali untuk memulai analisis.")