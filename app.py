import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import io
try:
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    import graphviz
except ImportError:
    st.error("⚠️ Pustaka sistem belum lengkap. Pastikan file requirements.txt & packages.txt sudah ada.")

# --- 1. CONFIG & ENGINE ---
st.set_page_config(page_title="Q1 SEM Multi-Variable Pro", layout="wide", page_icon="🎓")

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    st.error("❌ API Key Gemini tidak ditemukan di Secrets.")
    st.stop()

# --- 2. GENERATE DUMMY DATA (MULTI-VAR) ---
def generate_multi_var_dummy():
    rows = 70
    data = {}
    for i in range(1, 6):
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

# --- 3. MULTI-VARIABLE ANALYTICS ---
def analyze_multi_path(df, x_list, m_list, y_list):
    all_vars = x_list + m_list + y_list
    avg_scores = pd.DataFrame()
    q_reports = []
    
    for var in all_vars:
        if var:
            cols = [c for c in df.columns if c.startswith(var)]
            if cols:
                avg_scores[var] = df[cols].mean(axis=1)
                k = len(cols)
                alpha = (k/(k-1)) * (1-(df[cols].var().sum()/df[cols].sum(axis=1).var())) if k > 1 else 1.0
                q_reports.append({"Variable": var, "Alpha": round(alpha, 3)})

    for var in avg_scores.columns:
        target = avg_scores[var]
        feat = avg_scores.drop(columns=[var])
        if not feat.empty:
            r2 = LinearRegression().fit(feat, target).score(feat, target)
            vif = 1/(1-r2) if r2 < 1 else 10
            for item in q_reports:
                if item["Variable"] == var: item["VIF"] = round(vif, 3)
                
    return pd.DataFrame(q_reports), avg_scores

# --- 4. UI LAYOUT ---
st.image("https://i.ibb.co.com/23N3kpBY/Logo-DLI.png", width=160)
st.title("🎓 Q1 SEM Ultimate: Full Structural Model")

with st.sidebar:
    st.image("https://i.ibb.co.com/23N3kpBY/Logo-DLI.png", use_container_width=True)
    st.header("📂 Data Center")
    st.download_button("📥 Download Multi-Var Template", generate_multi_var_dummy(), "template_full_sem.xlsx")
    st.divider()
    uploaded_file = st.file_uploader("Unggah Data Riset (.xlsx)", type=["xlsx"])
    st.caption("Standard: Digital Learning Institute - 2026")

if uploaded_file:
    df = pd.read_excel(uploaded_file).ffill().bfill()
    
    st.subheader("1. Konfigurasi Multi-Variabel")
    col_x, col_m, col_y = st.columns(3)
    xs = [col_x.text_input(f"Variabel X{i}", "X1" if i==1 else "") for i in range(1, 6)]
    ms = [col_m.text_input(f"Variabel M{i}", "M1" if i==1 else "") for i in range(1, 6)]
    ys = [col_y.text_input(f"Variabel Y{i}", "Y1" if i==1 else "") for i in range(1, 6)]
    
    active_x = [x for x in xs if x]
    active_m = [m for m in ms if m]
    active_y = [y for y in ys if y]
    
    if active_x and active_y:
        q_df, df_avg = analyze_multi_path(df, active_x, active_m, active_y)
        
        st.subheader("2. Report Kualitas Konstruk")
        st.dataframe(q_df, use_container_width=True)
        
        st.subheader("3. Visualisasi Jalur (Direct & Indirect Effects)")
        dot = graphviz.Digraph(engine='dot')
        dot.attr(rankdir='LR', nodesep='0.4', ranksep='1.2')
        
        # Nodes
        for x in active_x: dot.node(x, x, shape='ellipse', style='filled', color='#E1F5FE')
        for m in active_m: dot.node(m, m, shape='ellipse', style='filled', color='#E8F5E9')
        for y in active_y: dot.node(y, y, shape='ellipse', style='filled', color='#FFFDE7')
        
        # 1. Jalur X -> M (Path a)
        for x in active_x:
            for m in active_m:
                slope, _, _, _, _ = stats.linregress(df_avg[x], df_avg[m])
                dot.edge(x, m, label=f"a={round(slope,2)}", color='#1565C0')

        # 2. Jalur M -> Y (Path b) & X -> Y (Direct Effect / Path c')
        for y in active_y:
            # Regresi Y terhadap X (semua) dan M (semua) untuk mendapatkan koefisien parsial
            predictors = active_x + active_m
            reg = LinearRegression().fit(df_avg[predictors], df_avg[y])
            coefs = dict(zip(predictors, reg.coef_))
            
            # Draw Path b (M -> Y)
            for m in active_m:
                dot.edge(m, y, label=f"b={round(coefs[m],2)}", color='#2E7D32')
            
            # Draw Direct Effect (X -> Y) - Garis Putus-putus Kuning
            for x in active_x:
                dot.edge(x, y, label=f"c'={round(coefs[x],2)}", style='dashed', color='#F9A825')

        st.graphviz_chart(dot)
        st.caption("Keterangan: Garis biru (a), garis hijau (b), garis putus-putus kuning (c' / Efek Langsung).")
        
        if st.button("🚀 GENERATE MULTI-PATH INTERPRETATION"):
            with st.spinner("AI sedang menganalisis efek langsung vs mediasi..."):
                summary_stats = df_avg.corr().to_string()
                prompt = f"Analisis mediasi untuk X:{active_x}, M:{active_m}, dan Y:{active_y}. Berikan penjelasan mengenai Efek Langsung (Direct Effect) dan Efek Tidak Langsung (Indirect Effect) berdasarkan korelasi: {summary_stats}. Sebutkan apakah terjadi Full atau Partial Mediation. Bahasa Indonesia akademik."
                st.write(model.generate_content(prompt).text)
    else:
        st.error("Minimal harus ada satu variabel X dan satu variabel Y.")

else:
    st.warning("Silakan unggah data atau gunakan template untuk memulai analisis.")

st.divider()
st.caption("Finalized Suite Ver 4.1 | Developed by Citra Kurniawan - 2026")