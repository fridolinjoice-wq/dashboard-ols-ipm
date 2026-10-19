import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

# ============ UBAH BAGIAN INI ============
KOTA = "Kota Bandung"          
FILE_DATA = "dataset_fiskal_97_hipoFull (1).csv"
Y_DEFAULT = "IPM"
X_DEFAULT = ["bm_pendidikan", "bm_kesehatan"]
# =========================================

st.set_page_config(page_title=f"Regresi OLS {KOTA}", layout="wide")

LABEL = {
    "bm_pendidikan": "Belanja Modal Pendidikan", "bm_kesehatan": "Belanja Modal Kesehatan",
    "bm_perumahan": "Belanja Modal Perumahan", "bm_lingkungan": "Belanja Modal Lingkungan",
    "bm_ekonomi": "Belanja Modal Ekonomi", "bm_infrastruktur": "Belanja Modal Infrastruktur",
    "bm_pariwisata": "Belanja Modal Pariwisata", "bm_perlindungan_sosial": "Belanja Modal Perlindungan Sosial",
    "IPM": "IPM", "KEMISKINAN": "Kemiskinan (%)", "PENGANGGURAN": "Pengangguran (%)",
    "KETIMPANGAN": "Ketimpangan", "PE": "Pertumbuhan Ekonomi (%)",
}
lab = lambda c: LABEL.get(c, c)


@st.cache_data
def load(path):
    return pd.read_csv(path)


df_all = load(FILE_DATA)
if KOTA not in set(df_all["Kota"]):
    st.error(f"Kota '{KOTA}' tidak ada di dataset. Cek penulisan nama kota di app.py.")
    st.stop()

df = df_all[df_all["Kota"] == KOTA].sort_values("Tahun").reset_index(drop=True)
Y_COLS = ["IPM", "KEMISKINAN", "PENGANGGURAN", "KETIMPANGAN", "PE"]
X_COLS = [c for c in df.columns if c not in ["Kota", "Tahun"] + Y_COLS]

# ---------- Sidebar ----------
st.sidebar.header("Pengaturan")
y_var = st.sidebar.selectbox("Variabel terikat (Y)", Y_COLS, index=Y_COLS.index(Y_DEFAULT), format_func=lab)
x_vars = st.sidebar.multiselect("Variabel bebas (X)", X_COLS, default=X_DEFAULT, format_func=lab)
alpha = st.sidebar.select_slider("Taraf signifikansi (α)", [0.01, 0.05, 0.10], value=0.05)
st.sidebar.markdown(f"**Dataset:** panel fiskal 97 kota Indonesia\n\n- {len(df_all):,} observasi\n- Studi kasus: **{KOTA}** ({len(df)} tahun)")

st.title(f"Analisis Regresi OLS: {KOTA}")
st.caption("Dashboard interaktif, Regresi Linear Berganda (Ordinary Least Squares) dan perbandingan model Machine Learning.")

if len(x_vars) < 1:
    st.warning("Pilih minimal 1 variabel bebas di sidebar.")
    st.stop()
if len(x_vars) > 4:
    st.warning("Hanya ada 11 observasi; pakai 4 variabel bebas atau kurang agar model tidak overfit.")

# ---------- Model OLS ----------
X = sm.add_constant(df[x_vars])
y = df[y_var]
model = sm.OLS(y, X).fit()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Ringkasan", "Tren", "Korelasi", "Regresi OLS", "Machine Learning"])

# ---------- Tab 1: Ringkasan ----------
with tab1:
    last = df.iloc[-1]
    c = st.columns(5)
    for col, k in zip(c, Y_COLS):
        col.metric(f"{lab(k)} ({int(last['Tahun'])})", f"{last[k]:.2f}",
                   f"{last[k] - df.iloc[-2][k]:+.2f} vs tahun lalu")
    st.subheader("Data")
    st.dataframe(df, use_container_width=True)
    st.subheader("Statistik deskriptif")
    st.dataframe(df.drop(columns=["Kota", "Tahun"]).describe().T.round(3), use_container_width=True)

# ---------- Tab 2: Tren ----------
with tab2:
    pilih = st.multiselect("Variabel yang ditampilkan", [y_var] + x_vars, default=[y_var] + x_vars[:1], format_func=lab)
    if pilih:
        long = df.melt(id_vars="Tahun", value_vars=pilih, var_name="Variabel", value_name="Nilai")
        long["Variabel"] = long["Variabel"].map(lab)
        st.plotly_chart(px.line(long, x="Tahun", y="Nilai", color="Variabel", markers=True), use_container_width=True)
        st.caption("Jika skalanya berbeda jauh, tampilkan satu variabel per grafik agar polanya terlihat.")

# ---------- Tab 3: Korelasi ----------
with tab3:
    cols = [y_var] + x_vars
    corr = df[cols].corr().round(3)
    corr.index = corr.columns = [lab(c) for c in cols]
    st.plotly_chart(px.imshow(corr, text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu_r"),
                    use_container_width=True)
    st.markdown("Korelasi antar variabel bebas yang sangat tinggi (|r| > 0,8) menandakan indikasi multikolinearitas.")

# ---------- Tab 4: Regresi OLS ----------
with tab4:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Observasi", int(model.nobs), f"tahun {df['Tahun'].min()}-{df['Tahun'].max()}")
    k2.metric("R-squared", f"{model.rsquared:.3f}", f"Adj. R² {model.rsquared_adj:.3f}")
    k3.metric("F-statistic", f"{model.fvalue:.3f}", f"p = {model.f_pvalue:.4f}")
    sig = int((model.pvalues[x_vars] < alpha).sum())
    k4.metric("Variabel signifikan", f"{sig} / {len(x_vars)}", f"pada α {int(alpha*100)}%")

    st.subheader("Model penelitian")
    y_tex = y_var.replace("_", r"\_")
    persamaan = f"\\widehat{{{y_tex}}} = {model.params['const']:.4f}"
    for i, v in enumerate(x_vars, 1):
        b = model.params[v]
        persamaan += f" {'+' if b >= 0 else '-'} {abs(b):.4f} \\cdot X_{{{i}}}"
    st.latex(persamaan)
    st.caption("  |  ".join(f"X{i} = {lab(v)}" for i, v in enumerate(x_vars, 1)))

    st.subheader("Hasil estimasi koefisien")
    ci = model.conf_int()
    tabel = pd.DataFrame({
        "Variabel": ["Konstanta"] + [lab(v) for v in x_vars],
        "Koefisien": model.params.values,
        "Std. Error": model.bse.values,
        "t-statistik": model.tvalues.values,
        "p-value": model.pvalues.values,
        "CI 95% bawah": ci[0].values,
        "CI 95% atas": ci[1].values,
    })
    tabel["Keputusan"] = np.where(tabel["p-value"] < alpha, "Signifikan", "Tidak signifikan")
    st.dataframe(tabel.round(4), use_container_width=True, hide_index=True)

    st.subheader("Uji asumsi klasik")
    from scipy import stats
    from statsmodels.stats.stattools import durbin_watson
    resid = model.resid
    sw_p = stats.shapiro(resid)[1]
    dw = durbin_watson(resid)
    a1, a2 = st.columns(2)
    a1.metric("Normalitas residual (Shapiro-Wilk)", f"p = {sw_p:.4f}", "normal" if sw_p > alpha else "tidak normal")
    a2.metric("Autokorelasi (Durbin-Watson)", f"{dw:.3f}", "mendekati 2 = aman")
    if len(x_vars) > 1:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif = pd.DataFrame({"Variabel": [lab(v) for v in x_vars],
                            "VIF": [variance_inflation_factor(X.values, i + 1) for i in range(len(x_vars))]})
        st.dataframe(vif.round(3), hide_index=True)
        st.caption("VIF di bawah 10 berarti tidak ada multikolinearitas serius.")

    st.subheader("Aktual vs prediksi")
    plot_df = pd.DataFrame({"Tahun": df["Tahun"], "Aktual": y, "Prediksi OLS": model.fittedvalues})
    st.plotly_chart(px.line(plot_df.melt("Tahun"), x="Tahun", y="value", color="variable", markers=True),
                    use_container_width=True)

    st.subheader("Interpretasi")
    ket = []
    ket.append(f"Secara simultan model {'signifikan' if model.f_pvalue < alpha else 'tidak signifikan'} "
               f"(F = {model.fvalue:.3f}, p = {model.f_pvalue:.4f}). Variabel bebas menjelaskan "
               f"{model.rsquared*100:.1f}% variasi {lab(y_var)} di {KOTA}.")
    for v in x_vars:
        b, p = model.params[v], model.pvalues[v]
        arah = "meningkatkan" if b > 0 else "menurunkan"
        if p < alpha:
            ket.append(f"- **{lab(v)}** berpengaruh signifikan (p = {p:.4f}): kenaikan 1 satuan {arah} {lab(y_var)} sebesar {abs(b):.4f}, ceteris paribus.")
        else:
            ket.append(f"- **{lab(v)}** tidak berpengaruh signifikan (p = {p:.4f}).")
    ket.append("\n*Catatan: hanya 11 observasi, jadi hasil ini bersifat indikatif dan hubungan yang ditemukan adalah asosiasi, bukan bukti sebab-akibat.*")
    st.markdown("\n".join(ket))

# ---------- Tab 5: Machine Learning ----------
with tab5:
    st.markdown("Dengan hanya 11 observasi, split train/test biasa tidak andal. Evaluasi memakai "
                "**Leave-One-Out Cross-Validation (LOOCV)**: tiap tahun bergantian menjadi data uji.")
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=3, random_state=42),
    }
    hasil, preds = [], {"Tahun": df["Tahun"], "Aktual": y}
    for nama, m in models.items():
        p = cross_val_predict(m, df[x_vars], y, cv=LeaveOneOut())
        hasil.append({"Model": nama, "R² (LOOCV)": r2_score(y, p),
                      "RMSE (LOOCV)": np.sqrt(mean_squared_error(y, p))})
        preds[nama] = p
    st.dataframe(pd.DataFrame(hasil).round(4), hide_index=True, use_container_width=True)
    st.plotly_chart(px.line(pd.DataFrame(preds).melt("Tahun"), x="Tahun", y="value", color="variable", markers=True),
                    use_container_width=True)
    st.caption("R² LOOCV bisa negatif; artinya model lebih buruk daripada sekadar menebak rata-rata. "
               "Itu hal wajar pada data sekecil ini dan layak dibahas di laporanmu.")
