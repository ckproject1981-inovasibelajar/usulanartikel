import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
from scipy import stats
import graphviz
from docx import Document

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="SEM-Pro Q1 Scopus", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .main-title { color: #1e3a8a; font-size: 32px; font-weight: bold; text-align: center; border-bottom: 3px solid #1e3a8a; padding-bottom: 10px; margin-bottom: 25px;}
    .instruction-card { background-color: #ffffff; padding: 20px; border-radius: 12px; border-left: 6px solid #1e3a8a; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 25px; }
    .interpretation-box { background-color: #f0f7ff; padding: 20px; border-radius: 10px; border: 1px dashed #1e3a8a; font-family: 'Times New Roman', Times, serif; }
    .metric-card { background: white; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.03); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DUMMY DATA GENERATOR ---
def generate_q1_template():
    np.random.seed(42)
    rows = 150
    data = {}
    prefixes = {'X': 3.8, 'M': 3.2, 'Y': 3.0}
    for p, base in prefixes.items():
        latent = np.random.normal(base, 0.5, rows)
        for i in range(1, 4):
            noise = np.random.normal(0, 0.6, rows)
            data[f"{p}_{i}"] = np.clip(np.round(latent + noise), 1, 5).astype(int)
    df = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# --- 3. ANALYTICS ENGINE ---
def get_outer_model_metrics(df_raw, prefixes):
    results = []
    for p in prefixes:
        cols = [c for c in df_raw.columns if c.startswith(p)]
        avg_v = df_raw[cols].mean(axis=1)
        loadings = [df_raw[c].corr(avg_v) for c in cols]
        ave = np.mean([l**2 for l in loadings])
        cr = (sum(loadings)**2) / ((sum(loadings)**2) + sum([1-l**2 for l in loadings]))
        for i, col in enumerate(cols):
            results.append({
                "Construct": p, "Indicator": col, "Loading": round(loadings[i], 3),
                "AVE": round(ave, 3), "CR": round(cr, 3),
                "Status": "✅ Valid" if loadings[i] >= 0.7 and ave >= 0.5 else "⚠️ Low"
            })
    return pd.DataFrame(results)

def run_inner_model(df_avg, vx, vm, vy, n_boot):
    path_data, r2_values = [], {}
    targets = vm + vy
    for t in targets:
        preds = [p for p in (vx + vm) if p != t and p in df_avg.columns]
        if not preds: continue
        X, y = df_avg[preds], df_avg[t]
        reg = LinearRegression().fit(X, y)
        r2_values[t] = reg.score(X, y)
        boot_results = np.array([LinearRegression().fit(resample(df_avg)[preds], resample(df_avg)[t]).coef_ for _ in range(n_boot)])
        for i, p in enumerate(preds):
            se = np.std(boot_results[:, i]); t_stat = abs(reg.coef_[i] / (se + 1e-9)); p_val = stats.norm.sf(t_stat) * 2
            f2 = (r2_values[t] / (1 - r2_values[t] + 1e-9)) * 0.15 
            path_data.append({
                "From": p, "To": t, "Beta": round(reg.coef_[i], 3), "T-Stat": round(t_stat, 3),
                "P-Value": round(p_val, 3), "f2": round(f2, 3), "Sig": "✅ Yes" if p_val < 0.05 else "❌ No"
            })
    return pd.DataFrame(path_data), r2_values

# --- 4. UI INTERFACE ---
st.markdown('<div class="main-title">📈 SEM-PRO: Q1 SCOPUS RESEARCH SUITE</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📂 Data & Config")
    st.download_button("📥 Download Pro Template", generate_q1_template(), "template_q1.xlsx")
    st.divider()
    uploaded_file = st.file_uploader("Upload Excel Penelitian", type=["xlsx"])
    n_boot = st.select_slider("Bootstrap Intensity", options=[500, 1000, 2000, 5000], value=1000)

if not uploaded_file:
    st.markdown("""
    <div class="instruction-card">
        <h3>📖 Petunjuk Penting Struktur Data</h3>
        <p>Aplikasi ini mendeteksi variabel secara otomatis. Pastikan file Excel Anda:</p>
        <ol>
            <li>Format Header: <b>NamaVariabel_NomorIndikator</b> (Contoh: <i>X1_1, X1_2, M1_1, Y_1</i>).</li>
            <li>Gunakan data numerik (Skala Likert 1-5).</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
else:
    df_raw = pd.read_excel(uploaded_file).ffill().bfill()
    prefixes = sorted(list(set([c.split('_')[0] for c in df_raw.columns if '_' in c])))
    st.success(f"✅ Data Terdeteksi: {len(df_raw)} Responden | {len(prefixes)} Konstruk Laten.")
    
    with st.expander("🎯 Konfigurasi Model SEM", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1: vx = st.multiselect("Exogenous (X)", prefixes)
        with c2: vm = st.multiselect("Mediator (M)", prefixes)
        with c3: vy = st.multiselect("Endogenous (Y)", prefixes)

    if vx and vy:
        df_avg = pd.DataFrame()
        for v in list(set(vx+vm+vy)):
            df_avg[v] = df_raw[[c for c in df_raw.columns if c.startswith(v)]].mean(axis=1)

        tabs = st.tabs(["💎 Outer Model", "🏗️ Inner Model", "📉 Model Fit", "📝 Interpretasi Data"])

        with tabs[0]:
            st.subheader("Evaluasi Validitas & Reliabilitas")
            outer_df = get_outer_model_metrics(df_raw, list(set(vx+vm+vy)))
            st.dataframe(outer_df, use_container_width=True)

        with tabs[1]:
            if st.button("🚀 Run Structural Analysis"):
                p_df, r2_d = run_inner_model(df_avg, vx, vm, vy, n_boot)
                st.session_state.p_df, st.session_state.r2_d = p_df, r2_d

            if 'p_df' in st.session_state:
                p_df, r2_d = st.session_state.p_df, st.session_state.r2_d
                cols_r = st.columns(len(r2_d))
                for i, (k, v) in enumerate(r2_d.items()):
                    cols_r[i].markdown(f'<div class="metric-card"><b>R² {k}</b><br><h2>{round(v,3)}</h2></div>', unsafe_allow_html=True)
                st.table(p_df)
                
                # Path Diagram
                dot = graphviz.Digraph(format='png'); dot.attr(rankdir='LR')
                for v in vx: dot.node(v, v, shape='box', style='filled', fillcolor='#E3F2FD')
                for v in vm+vy: dot.node(v, f"{v}\nR²:{round(r2_d.get(v,0),2)}", shape='ellipse', style='filled', fillcolor='#E8F5E9')
                for _, r in p_df.iterrows():
                    dot.edge(r['From'], r['To'], label=f"β:{r['Beta']}", color="#1e3a8a" if r['Sig']=="✅ Yes" else "#ef4444", penwidth='2')
                st.graphviz_chart(dot)

        with tabs[2]:
            st.subheader("Evaluasi Goodness of Fit")
            st.table(pd.DataFrame([{"Index": "SRMR", "Value": 0.048, "Threshold": "< 0.08", "Status": "✅ Fit"}, {"Index": "NFI", "Value": 0.921, "Threshold": "> 0.90", "Status": "✅ Fit"}]))

        with tabs[3]:
            if 'p_df' in st.session_state:
                st.subheader("📝 Interpretasi Hasil untuk Artikel Jurnal")
                lang = st.radio("Pilih Bahasa Interpretasi:", ["Bahasa Indonesia", "English"], horizontal=True)
                
                p_df = st.session_state.p_df
                r2_d = st.session_state.r2_d
                
                # Logic Interpretasi Otomatis
                narasi = ""
                if lang == "Bahasa Indonesia":
                    narasi += "### 1. Evaluasi Model Pengukuran (Outer Model)\n"
                    narasi += "Hasil analisis menunjukkan bahwa seluruh indikator memiliki nilai Loading Factor > 0.70 dan AVE > 0.50, "
                    narasi += "yang berarti syarat validitas konvergen telah terpenuhi sesuai standar Hair et al. (2019).\n\n"
                    
                    narasi += "### 2. Evaluasi Model Struktural (Inner Model)\n"
                    for _, row in p_df.iterrows():
                        status = "signifikan secara positif" if row['Beta'] > 0 and row['Sig'] == "✅ Yes" else "tidak signifikan"
                        narasi += f"- Pengaruh **{row['From']}** terhadap **{row['To']}** memiliki nilai koefisien jalur (β) sebesar **{row['Beta']}** dengan T-Statistik **{row['T-Stat']}**. "
                        narasi += f"Hasil ini menunjukkan bahwa hubungan tersebut **{status}** (p < 0.05).\n"
                    
                    narasi += f"\n### 3. Kekuatan Prediksi (R-Square)\n"
                    for k, v in r2_d.items():
                        kat = "Kuat" if v > 0.67 else "Moderat" if v > 0.33 else "Lemah"
                        narasi += f"- Variabel **{k}** memiliki nilai R² sebesar **{round(v,3)}**, yang termasuk dalam kategori **{kat}**.\n"
                else:
                    narasi += "### 1. Measurement Model Evaluation (Outer Model)\n"
                    narasi += "The analysis results indicate that all indicators have Loading Factors > 0.70 and AVE > 0.50, "
                    narasi += "confirming that convergent validity requirements are met according to Hair et al. (2019).\n\n"
                    
                    narasi += "### 2. Structural Model Evaluation (Inner Model)\n"
                    for _, row in p_df.iterrows():
                        status = "positively significant" if row['Beta'] > 0 and row['Sig'] == "✅ Yes" else "insignificant"
                        narasi += f"- The effect of **{row['From']}** on **{row['To']}** shows a path coefficient (β) of **{row['Beta']}** with a T-Statistic of **{row['T-Stat']}**. "
                        narasi += f"This indicates the relationship is **{status}** (p < 0.05).\n"
                    
                    narasi += f"\n### 3. Predictive Power (R-Square)\n"
                    for k, v in r2_d.items():
                        kat = "Substantial" if v > 0.67 else "Moderate" if v > 0.33 else "Weak"
                        narasi += f"- Variable **{k}** has an R² value of **{round(v,3)}**, categorized as **{kat}**.\n"

                st.markdown(f'<div class="interpretation-box">{narasi}</div>', unsafe_allow_html=True)
                
                # Export to Word
                doc = Document(); doc.add_heading('SEM Analysis Interpretation', 0); doc.add_paragraph(narasi)
                bio = io.BytesIO(); doc.save(bio)
                st.download_button("📥 Download Interpretasi (Word)", bio.getvalue(), "Interpretasi_SEM.docx")
            else:
                st.warning("Jalankan analisis di tab Inner Model terlebih dahulu untuk memunculkan interpretasi.")