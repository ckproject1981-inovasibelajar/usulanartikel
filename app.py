import streamlit as st
import pandas as pd
import numpy as np
import io
import graphviz
from semopy import Model, calc_stats

# --- CONFIG ---
st.set_page_config(page_title="SEM Pro Assistant v3.3", layout="wide")

# --- GEN DATA (Ultra Stable) ---
def generate_stable_data(nx, nm, ny):
    rows = 500
    data = {}
    base = np.random.normal(3, 0.5, rows)
    
    def add_vars(prefix, count, latent_base):
        for i in range(1, count + 1):
            l_val = latent_base + np.random.normal(0, 0.1, rows)
            for j in range(1, 4): # 3 Indikator
                data[f"{prefix}{i}_{j}"] = np.clip(0.9 * l_val + np.random.normal(0, 0.05, rows), 1, 5)

    add_vars("X", nx, base)
    add_vars("M", nm, 0.8 * base)
    add_vars("Y", ny, 0.7 * base)
    
    df = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🔬 SEM Engine v3.3")
    with st.expander("🛠️ Control Panel"):
        nx_i = st.number_input("Variabel X", 1, 5, 3)
        nm_i = st.number_input("Variabel M", 0, 5, 2)
        ny_i = st.number_input("Variabel Y", 1, 5, 2)
        st.download_button("📥 Get Template", generate_stable_data(nx_i, nm_i, ny_i), "data_sem.xlsx")
    uploaded = st.file_uploader("Upload Data", type=["xlsx"])

# --- MAIN ---
if uploaded:
    # 1. Loading & Cleaning (Strikter)
    df = pd.read_excel(uploaded)
    df.columns = [str(c).strip().replace(' ', '') for c in df.columns] # Hapus spasi tak terlihat
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    
    # 2. Variable Detection
    available = sorted(list(set([c.split('_')[0] for c in df.columns if '_' in c])))
    
    st.header("📐 Model Setup")
    c1, c2, c3 = st.columns(3)
    with c1: vx = st.multiselect("Exogenous (X)", available)
    with c2: vm = st.multiselect("Mediators (M)", available)
    with c3: vy = st.multiselect("Endogenous (Y)", available)

    if vx and vy:
        # 3. Dynamic Syntax Building
        m_syntax = ""
        for v in (vx + vm + vy):
            inds = [c for c in df.columns if c.startswith(v + "_")]
            if inds: m_syntax += f"{v} =~ {' + '.join(inds)}\n"
        
        s_syntax = ""
        for m in vm:
            for x in vx: s_syntax += f"{m} ~ {x}\n"
        for y in vy:
            for m in (vx + vm): s_syntax += f"{y} ~ {m}\n"

        if st.button("🏁 Run Advanced Analysis"):
            try:
                # DEBUG: Tampilkan syntax yang dikirim ke SEMOPY
                with st.expander("🔍 Lihat Syntax Model"):
                    st.code(m_syntax + s_syntax)
                
                model = Model(m_syntax + s_syntax)
                model.fit(df)
                
                # Cek Inspect secara manual sebelum calc_stats
                inspected = model.inspect()
                
                try:
                    stats_res = calc_stats(model).T
                    
                    st.success("✅ Analisis Berhasil")
                    m1, m2, m3 = st.columns(3)
                    # Mengambil nilai baris pertama (index 0) secara aman
                    m1.metric("CFI", f"{stats_res.iloc[0]['CFI']:.3f}")
                    m2.metric("RMSEA", f"{stats_res.iloc[0]['RMSEA']:.3f}")
                    m3.metric("SRMR", f"{stats_res.iloc[0]['SRMR']:.3f}")
                    
                    st.write("### Tabel Koefisien")
                    st.dataframe(inspected)
                except Exception as e_stats:
                    st.error(f"⚠️ Model konvergen tapi statistik gagal dihitung: {e_stats}")
                    st.write("Coba periksa apakah jumlah data cukup banyak (min. 100-200 baris).")

            except Exception as e:
                st.error(f"❌ Gagal Total: {e}")
                st.info("Saran: Kurangi jumlah variabel mediator atau pastikan nama kolom di Excel persis X1_1, X1_2, dst.")