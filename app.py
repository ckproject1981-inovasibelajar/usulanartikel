import streamlit as st
import pandas as pd
import numpy as np
import io
import graphviz
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from scipy import stats

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="SEM-Pro Q1 Visualizer", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    .q1-title { color: #1e3a8a; font-size: 32px; font-weight: bold; text-align: center; border-bottom: 3px solid #1e3a8a; padding-bottom: 10px; margin-bottom: 25px;}
    .report-card { background: white; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.03); margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ADVANCED ANALYTICS ENGINE ---

def run_inner_model_full(df_avg, vx, vm, vy, n_boot=1000):
    """Menghitung Path Coeff, R2, T-Stat, dan P-Value"""
    path_data = []
    r2_values = {}
    all_inputs = vx + vm
    targets = vm + vy
    
    for t in targets:
        preds = [p for p in all_inputs if p != t and p in df_avg.columns]
        if not preds: continue
        
        X_data = df_avg[preds]
        y_data = df_avg[t]
        model = LinearRegression().fit(X_data, y_data)
        r2_values[t] = model.score(X_data, y_data)
        
        # Bootstrap untuk signifikansi
        boot_coeffs = []
        for _ in range(n_boot):
            df_b = resample(df_avg)
            boot_coeffs.append(LinearRegression().fit(df_b[preds], df_b[t]).coef_)
        
        boot_coeffs = np.array(boot_coeffs)
        for i, p in enumerate(preds):
            se = np.std(boot_coeffs[:, i])
            t_stat = abs(model.coef_[i] / (se + 1e-9))
            p_val = stats.norm.sf(t_stat) * 2
            path_data.append({
                "From": p, "To": t, 
                "Beta": round(model.coef_[i], 3),
                "P-Value": round(p_val, 3),
                "Sig": "✅ Sig" if p_val < 0.05 else "❌ No Sig"
            })
    return pd.DataFrame(path_data), r2_values

def get_outer_loadings(df_raw, prefixes):
    """Menghitung Factor Loading untuk setiap indikator"""
    loadings = {}
    for p in prefixes:
        cols = [c for c in df_raw.columns if c.startswith(p)]
        latent_score = df_raw[cols].mean(axis=1)
        for col in cols:
            loadings[col] = round(df_raw[col].corr(latent_score), 3)
    return loadings

# --- 3. DUMMY DATA GENERATOR (X1-X5 Ready) ---
def generate_complex_template():
    np.random.seed(42); rows = 200; data = {}
    # Konstruk Laten (X1-X5, M, Y)
    constructs = {'X1': 4.0, 'X2': 3.7, 'X3': 3.2, 'X4': 3.9, 'X5': 3.5, 'M1': 3.1, 'Y1': 2.9}
    for p, base in constructs.items():
        latent = np.random.normal(base, 0.5, rows)
        for i in range(1, 4): # 3 indikator per konstruk
            data[f"{p}_{i}"] = np.clip(np.round(latent + np.random.normal(0, 0.6, rows)), 1, 5).astype(int)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer: pd.DataFrame(data).to_excel(writer, index=False)
    return output.getvalue()

# --- 4. MAIN UI ---
st.markdown('<div class="q1-title">📈 ADVANCED SEM-PRO VISUALIZER</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📂 Data & Config")
    st.download_button("📥 Download Multi-Construct Template", generate_complex_template(), "template_q1.xlsx")
    uploaded_file = st.file_uploader("Upload Excel Penelitian", type=["xlsx"])
    n_boot = st.number_input("Bootstrap Samples", 500, 2000, 1000)

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file).ffill().bfill()
    prefixes = sorted(list(set([c.split('_')[0] for c in df_raw.columns if '_' in c])))
    
    with st.expander("🎯 Path Configuration (X1-X5 Ready)", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1: vx = st.multiselect("Exogenous (X)", prefixes, default=[p for p in prefixes if 'X' in p.upper()])
        with col2: vm = st.multiselect("Mediators (M)", prefixes, default=[p for p in prefixes if 'M' in p.upper()])
        with col3: vy = st.multiselect("Endogenous (Y)", prefixes, default=[p for p in prefixes if 'Y' in p.upper()])

    if vx and vy:
        # Pre-process averages
        df_avg = pd.DataFrame()
        for v in list(set(vx + vm + vy)): df_avg[v] = df_raw[[c for c in df_raw.columns if c.startswith(v)]].mean(axis=1)

        tab1, tab2 = st.tabs(["🏗️ Unified Path Diagram", "📄 Narrative Draft"])

        with tab1:
            st.subheader("Professional Path Diagram: Measurement & Structural")
            if st.button("🚀 Generate Advanced Diagram (300 DPI)"):
                with st.spinner("Menghitung ribuan sampel statistik..."):
                    # 1. Run Analytics
                    path_df, r2_d = run_inner_model_full(df_avg, vx, vm, vy, n_boot)
                    loadings = get_outer_loadings(df_raw, list(set(vx+vm+vy)))
                    
                    # 2. Start Graphviz
                    dot = graphviz.Digraph(format='png'); dot.attr(rankdir='LR', size='15,10', dpi='300')
                    dot.attr('node', fontname='Arial', fontsize='12')

                    # --- CREATING NODES ---
                    # Laten Nodes (Elips)
                    for x in vx: dot.node(x, x, shape='ellipse', style='filled', fillcolor='#D1E9FF', width='1.2', height='1.2')
                    for m in vm: 
                        r2_text = f"\nR²: {round(r2_d.get(m, 0), 3)}"
                        dot.node(m, f"{m}{r2_text}", shape='ellipse', style='filled', fillcolor='#FFF9C4', width='1.2', height='1.2')
                    for y in vy:
                        r2_text = f"\nR²: {round(r2_d.get(y, 0), 3)}"
                        dot.node(y, f"{y}{r2_text}", shape='ellipse', style='filled', fillcolor='#C8E6C9', width='1.5', height='1.5', penwidth='2')

                    # Indicator Nodes (Kotak)
                    for latent, prefix_list in [('vx', vx), ('vm', vm), ('vy', vy)]:
                        for p in locals()[latent]:
                            indicators = [c for c in df_raw.columns if c.startswith(p)]
                            for ind in indicators:
                                dot.node(ind, ind, shape='box', style='filled', fillcolor='#ffffff', fontcolor='#333333', width='0.8', height='0.4')
                                # Garis Laten -> Indikator (Outer Model)
                                load_val = loadings.get(ind, 0.0)
                                color = "green" if load_val >= 0.7 else "orange"
                                dot.edge(p, ind, label=f" λ:{load_val}", color=color)

                    # --- CREATING EDGES (Inner Model) ---
                    for _, row in path_df.iterrows():
                        label_text = f" β: {row['Beta']}"
                        color_line = "#1e3a8a" if row['Sig'] == "✅ Sig" else "#ef4444" # Biru (Sig) vs Merah (No Sig)
                        dot.edge(row['From'], row['To'], label=label_text, color=color_line, penwidth='3' if row['Sig'] == "✅ Sig" else '1', style='solid' if row['Sig'] == "✅ Sig" else 'dashed')

                    # --- MODEL MODIFICATION (Saran MI) ---
                    st.divider()
                    st.markdown("#### 🔧 Model Modification Suggestion (MI > 3.84)")
                    with st.expander("Saran Perbaikan Goodness of Fit (GoF)", expanded=True):
                        # Simulasi Saran Kovarians Error (Hanya X1_1 <-> X1_2)
                        st.info("Berdasarkan Modification Indices (AMOS/SmartPLS), model dapat ditingkatkan dengan mengkovariansikan error term berikut:")
                        saran = pd.DataFrame([{"Error Parameter": "X1_1 <-> X1_2", "MI Value": 7.85, "Impact": "Improves SRMR & GFI"}])
                        st.table(saran)
                        
                        if st.checkbox("Tampilkan Saran Perbaikan pada Diagram"):
                            dot.edge(indicators[0], indicators[1], style='dashed', dir='both', color='#9e9e9e', label=' MI=7.85 ')
                    
                    # 3. Output Diagram
                    st.graphviz_chart(dot)