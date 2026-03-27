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
    st.error("⚠️ Pustaka belum lengkap. Pastikan file requirements.txt dan packages.txt sudah ada di GitHub.")

# --- 1. CONFIG & ENGINE ---
st.set_page_config(page_title="Q1 SEM Ultimate Pro", layout="wide", page_icon="🎓")

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    # Inisialisasi Model Gemini
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    st.error("❌ API Key missing!")
    st.stop()

# --- 2. FUNGSI GENERATE DUMMY EXCEL ---
def generate_dummy_excel():
    rows = 50
    data = {}
    # Membuat korelasi buatan agar hasil statistik bagus
    base_x = np.random.randint(3, 6, rows)
    for i in range(1, 6):
        data[f'X{i}'] = np.clip(base_x + np.random.normal(0, 0.5, rows), 1, 5).round()
    
    base_m = base_x * 0.7 + np.random.normal(0, 0.5, rows)
    for i in range(1, 6):
        data[f'M{i}'] = np.clip(base_m + np.random.normal(0, 0.5, rows), 1, 5).round()
        
    base_y = base_m * 0.8 + np.random.normal(0, 0.5, rows)
    for i in range(1, 6):
        data[f'Y{i}'] = np.clip(base_y + np.random.normal(0, 0.5, rows), 1, 5).round()
    
    df_dummy = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_dummy.to_excel(writer, index=False)
    return output.getvalue()

# --- 3. CORE ANALYTICS ---
def run_analysis(df, x_code, m_code, y_code):
    codes = [x_code, m_code, y_code]
    avg_scores = pd.DataFrame()
    q_data = []
    
    for c in codes:
        cols = [col for col in df.columns if col.startswith(c)]
        if cols:
            avg_scores[c] = df[cols].mean(axis=1)
            # Alpha
            k = len(cols)
            alpha = (k/(k-1)) * (1-(df[cols].var().sum()/df[cols].sum(axis=1).var())) if k > 1 else 1.0
            q_data.append({"Variable": c, "Alpha": round(alpha, 3)})

    # VIF
    for c in avg_scores.columns:
        target = avg_scores[c]
        feat = avg_scores.drop(columns=[c])
        r2 = LinearRegression().fit(feat, target).score(feat, target)
        vif = 1/(1-r2) if r2 < 1 else 10
        next(i for i in q_data if i["Variable"] == c)["VIF"] = round(vif, 3)

    # Path
    slope_a, _, _, _, _ = stats.linregress(avg_scores[x_code], avg_scores[m_code])
    reg_final = LinearRegression().fit(avg_scores[[x_code, m_code]], avg_scores[y_code])
    path_b, path_c_p = reg_final.coef_[1], reg_final.coef_[0]
    
    return pd.DataFrame(q_data), {
        "a": round(slope_a, 3), "b": round(path_b, 3), "c_prime": round(path_c_p, 3),
        "ind": round(slope_a * path_b, 3), "tot": round(path_c_p + (slope_a * path_b), 3)
    }

# --- 4. UI LAYOUT ---
st.image("https://i.ibb.co.com/23N3kpBY/Logo-DLI.png", width=160)
st.title("🎓 Q1 SEM Ultimate: Analytics & Visualization")

with st.sidebar:
    st.image("https://i.ibb.co.com/23N3kpBY/Logo-DLI.png", use_container_width=True)
    st.header("📂 Data Center")
    
    # Tombol Download Dummy
    st.write("Belum punya data?")
    dummy_file = generate_dummy_excel()
    st.download_button("📥 Download Contoh Excel", dummy_file, "dummy_data_sem.xlsx")
    
    st.divider()
    uploaded_file = st.file_uploader("Unggah File Anda (.xlsx)", type=["xlsx"])

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file)
    df = df_raw.fillna(df_raw.mean(numeric_only=True))
    
    st.subheader("1. Konfigurasi Variabel")
    st.info("Petunjuk: Pastikan nama kolom di Excel diawali dengan huruf variabel (contoh: X1, X2... M1, M2...)")
    
    col1, col2, col3 = st.columns(3)
    vx = col1.text_input("Variabel X", "X")
    vm = col2.text_input("Variabel M", "M")
    vy = col3.text_input("Variabel Y", "Y")
    
    q_df, res = run_analysis(df, vx, vm, vy)
    
    st.subheader("2. Reliability (Alpha) & Collinearity (VIF)")
    st.table(q_df)
    
    st.subheader("3. Path Analysis & Diagram")
    c_vis, c_stat = st.columns([2, 1])
    
    with c_vis:
        dot = graphviz.Digraph(engine='dot')
        dot.attr(rankdir='LR')
        dot.node('X', vx, shape='ellipse', style='filled', color='#E1F5FE')
        dot.node('M', vm, shape='ellipse', style='filled', color='#E8F5E9')
        dot.node('Y', vy, shape='ellipse', style='filled', color='#FFFDE7')
        dot.edge('X', 'M', label=f"a={res['a']}")
        dot.edge('M', 'Y', label=f"b={res['b']}")
        dot.edge('X', 'Y', label=f"c'={res['c_prime']}", style='dashed')
        st.graphviz_chart(dot)
        
    with c_stat:
        st.metric("Indirect Effect", res['ind'])
        st.metric("Total Effect", res['tot'])
        st.write(f"**Status:** {'Partial Mediation' if abs(res['c_prime']) > 0.1 else 'Full Mediation'}")

    if st.button("🚀 GENERATE MANUSCRIPT"):
        t1, t2 = st.tabs(["💡 Interpretasi", "📝 Draf IMRAD"])
        with t1:
            st.write(model.generate_content(f"Berikan interpretasi akademik bahasa Indonesia untuk mediasi {vx}->{vm}->{vy} dengan hasil a={res['a']}, b={res['b']}, c'={res['c_prime']}. Gunakan standar Nick Shryane.").text)
        with t2:
            st.code(model.generate_content(f"Write Scopus Q1 Results section in English for mediation {vx}->{vm}->{vy}. Include path coefficients: a={res['a']}, b={res['b']}, c'={res['c_prime']}.").text)

else:
    st.warning("👋 Selamat Datang! Silakan unduh contoh file di sidebar atau unggah file Excel Anda.")

st.divider()
st.caption("Finalized Suite Ver 3.2 | Distructive Learning Innovation - @Citra Kurniawan - 2026")