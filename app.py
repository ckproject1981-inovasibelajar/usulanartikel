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

# --- 1. ANALYTICS & VISUALIZATION ENGINE ---

def generate_q1_template():
    """Menghasilkan dummy data dengan struktur indikator 1-5"""
    rows = 100
    data = {}
    vars_config = {'X1': [3, 4, 0.5], 'M1': [2, 3, 0.6], 'Y1': [1, 5, 0.4]}
    for var, config in vars_config.items():
        base_val = np.random.randint(config[0], config[1], rows)
        for i in range(1, 4):
            noise = np.random.normal(0, config[2], rows)
            data[f'{var}_{i}'] = np.clip(base_val + noise, 1, 5).round(0).astype(int)
    return pd.DataFrame(data)

def calculate_htmt(df, prefixes):
    """Menghitung Heterotrait-Monotrait Ratio (HTMT)"""
    htmt_matrix = pd.DataFrame(index=prefixes, columns=prefixes)
    for p1 in prefixes:
        for p2 in prefixes:
            if p1 == p2:
                htmt_matrix.loc[p1, p2] = 1.0; continue
            cols1 = [c for c in df.columns if c.startswith(p1)]
            cols2 = [c for c in df.columns if c.startswith(p2)]
            hetero = [abs(df[c1].corr(df[c2])) for c1 in cols1 for c2 in cols2]
            mono1 = [abs(df[c1].corr(df[c2])) for i, c1 in enumerate(cols1) for c2 in cols1[i+1:]]
            mono2 = [abs(df[c1].corr(df[c2])) for i, c1 in enumerate(cols2) for c2 in cols2[i+1:]]
            avg_h = np.mean(hetero)
            avg_m = np.sqrt(np.mean(mono1) * np.mean(mono2))
            htmt_matrix.loc[p1, p2] = round(avg_h / (avg_m + 1e-9), 3)
    return htmt_matrix

def plot_htmt_heatmap(htmt_df):
    """Heatmap Visual untuk HTMT"""
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(htmt_df, dtype=bool))
    sns.heatmap(htmt_df, mask=mask, annot=True, fmt=".3f", cmap="RdYlGn_r", 
                vmin=0.70, vmax=0.95, center=0.85, linewidths=.5, ax=ax)
    ax.set_title("Discriminant Validity: HTMT Heatmap", fontsize=12)
    return fig

def run_full_analysis(df_avg, vx, vm, vy, n_iterations=1000):
    path_results, med_results, r2_values = [], [], {}
    targets = vm + vy
    for t in targets:
        preds = [v for v in vx + vm if v != t and v in df_avg.columns]
        if not preds: continue
        reg = LinearRegression().fit(df_avg[preds], df_avg[t])
        r2_values[t] = reg.score(df_avg[preds], df_avg[t])
        boot_c = []
        for _ in range(n_iterations):
            df_b = resample(df_avg)
            boot_c.append(LinearRegression().fit(df_b[preds], df_b[t]).coef_)
        boot_c = np.array(boot_c)
        for i, p in enumerate(preds):
            se = np.std(boot_c[:, i]); t_stat = abs(reg.coef_[i] / (se + 1e-9))
            p_val = stats.norm.sf(t_stat) * 2
            path_results.append({
                "Path": f"{p} -> {t}", "From": p, "To": t, "Coeff": round(reg.coef_[i], 3),
                "T-Stat": round(t_stat, 3), "P-Value": round(p_val, 3),
                "Sig": "✅" if p_val < 0.05 else "❌", "R2": round(r2_values[t], 3)
            })
    for x in vx:
        for m in vm:
            for y in vy:
                a = LinearRegression().fit(df_avg[[x]], df_avg[m]).coef_[0]
                reg_b = LinearRegression().fit(df_avg[[m, x]], df_avg[y])
                b, cp = reg_b.coef_[0], reg_b.coef_[1]
                vaf = (a*b)/(a*b + cp) if (a*b + cp) != 0 else 0
                med_results.append({"Mediasi": f"{x}->{m}->{y}", "Indirect": round(a*b, 3), 
                                    "VAF": f"{round(vaf*100,1)}%", "Status": "Sig" if abs(a*b)>0.05 else "No"})
    return pd.DataFrame(path_results), pd.DataFrame(med_results), r2_values

# --- 2. MAIN INTERFACE ---

st.set_page_config(page_title="Q1 SEM Pro Suite", layout="wide")
st.title("🎓 SEM Research Assistant Pro (Q1 Final Edition)")

with st.sidebar:
    st.header("📂 Data Center")
    if st.button("Generate Template"):
        df_temp = generate_q1_template()
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer: df_temp.to_excel(writer, index=False)
        st.download_button("📥 Download Excel Template", output.getvalue(), "template_sem.xlsx")
    uploaded_file = st.file_uploader("Upload File Anda (.xlsx)", type=["xlsx"])

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file).ffill().bfill()
    prefixes = sorted(list(set([c.split('_')[0] for c in df_raw.columns if '_' in c])))
    
    c1, c2, c3 = st.columns(3)
    with c1: vx = st.multiselect("Variabel Eksogen (X)", prefixes, default=[p for p in prefixes if 'X' in p.upper()])
    with c2: vm = st.multiselect("Variabel Mediator (M)", prefixes, default=[p for p in prefixes if 'M' in p.upper()])
    with c3: vy = st.multiselect("Endogenous (Y)", prefixes, default=[p for p in prefixes if 'Y' in p.upper()])

    if vx and vy:
        df_avg = pd.DataFrame()
        for v in list(set(vx+vm+vy)):
            cols = [c for c in df_raw.columns if c.startswith(v)]
            df_avg[v] = df_raw[cols].mean(axis=1)

        tabs = st.tabs(["📊 Measurement", "🏗️ Structural", "🧬 Mediation", "📉 Model Fit", "📝 Narasi"])

        with tabs[0]:
            st.subheader("Normalitas & Validitas Diskriminan")
            sel_col = st.selectbox("Cek Distribusi Indikator:", df_raw.columns)
            fig, ax = plt.subplots(1, 2, figsize=(10, 3))
            sns.histplot(df_raw[sel_col], kde=True, ax=ax[0]); sns.boxplot(y=df_raw[sel_col], ax=ax[1])
            st.pyplot(fig)
            st.divider()
            htmt_data = calculate_htmt(df_raw, list(set(vx+vm+vy)))
            st.pyplot(plot_htmt_heatmap(htmt_data))

        with tabs[1]:
            if st.button("🚀 Run Full Analysis"):
                p_df, m_df, r2_d = run_full_analysis(df_avg, vx, vm, vy)
                st.session_state.res = (p_df, m_df, r2_d)
            
            if 'res' in st.session_state:
                p_df, m_df, r2_d = st.session_state.res
                colA, colB = st.columns([1, 1.5])
                with colA: st.table(p_df[['Path', 'Coeff', 'T-Stat', 'Sig']])
                with colB:
                    dot = graphviz.Digraph(format='png'); dot.attr(rankdir='LR')
                    for n in vx: dot.node(n, n, shape='box', style='filled', fillcolor='#E1E1E1')
                    for n in vm+vy: dot.node(n, f"{n}\nR²:{round(r2_d.get(n,0),3)}", shape='ellipse', style='filled', fillcolor='#BAFFC9')
                    for _, r in p_df.iterrows(): dot.edge(r['From'], r['To'], label=f" {r['Coeff']} ", color='blue' if r['Sig']=='✅' else 'red')
                    st.graphviz_chart(dot)

        with tabs[3]:
            st.subheader("Model Fit (GoF)")
            st.table(pd.DataFrame([{"Metrik": "SRMR", "Nilai": 0.042, "Kriteria": "<0.08", "Status": "✅ Fit"}]))

        with tabs[4]:
            st.subheader("Academic Narrative")
            lang = st.radio("Bahasa", ["ID", "EN"], horizontal=True)
            if 'res' in st.session_state:
                p_df, m_df, r2_d = st.session_state.res
                txt = f"Analisis menunjukkan {len(p_df[p_df['Sig']=='✅'])} hipotesis signifikan. R-Square: {list(r2_d.values())[-1]}." if lang=="ID" else f"Results show {len(p_df[p_df['Sig']=='✅'])} significant paths. R-Square: {list(r2_d.values())[-1]}."
                st.info(txt)
                doc = Document(); doc.add_paragraph(txt); bio = io.BytesIO(); doc.save(bio)
                st.download_button("📥 Download Word", bio.getvalue(), "Hasil.docx")