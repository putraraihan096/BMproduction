import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cluster Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global Matplotlib Style ────────────────────────────────────────────────────
rcParams['font.family'] = 'monospace'
plt.rcParams.update({
    'axes.facecolor': '#0f1117',
    'figure.facecolor': '#0f1117',
    'axes.edgecolor': '#2a2d3a',
    'axes.labelcolor': '#8b93a7',
    'xtick.color': '#8b93a7',
    'ytick.color': '#8b93a7',
    'text.color': '#c9d1e0',
    'grid.color': '#1e2130',
    'grid.linewidth': 0.6,
})

# ── Design Tokens ──────────────────────────────────────────────────────────────
PALETTE = ["#6c8ebf", "#82b366", "#d6a35b", "#b85c5c", "#9a7fc7", "#5ba89a", "#c47fc4", "#7bb8c4"]
BG_DARK   = "#0f1117"
BG_CARD   = "#161b27"
BG_BORDER = "#252a3a"
TXT_PRI   = "#e8edf5"
TXT_SEC   = "#8b93a7"
ACCENT    = "#6c8ebf"

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: {BG_DARK};
    color: {TXT_PRI};
}}

section[data-testid="stSidebar"] {{
    background: #0b0e16 !important;
    border-right: 1px solid {BG_BORDER};
    padding-top: 1.5rem;
}}
section[data-testid="stSidebar"] .block-container {{
    padding-top: 0;
}}

.main .block-container {{
    padding: 2rem 2.5rem 3rem;
    max-width: 1400px;
}}

.dash-header {{
    display: flex;
    align-items: center;
    gap: 14px;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid {BG_BORDER};
    margin-bottom: 2rem;
}}
.dash-icon {{
    font-size: 28px;
    font-family: 'IBM Plex Mono', monospace;
    color: {ACCENT};
    letter-spacing: -2px;
}}
.dash-title {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: {TXT_PRI};
}}
.dash-subtitle {{
    font-size: 12px;
    color: {TXT_SEC};
    letter-spacing: 1px;
    margin-top: 2px;
}}

.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 1.5rem;
}}
.kpi-card {{
    background: {BG_CARD};
    border: 1px solid {BG_BORDER};
    border-radius: 8px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: {ACCENT};
    opacity: 0.6;
}}
.kpi-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: {TXT_SEC};
    margin-bottom: 8px;
}}
.kpi-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px;
    font-weight: 600;
    color: {TXT_PRI};
    line-height: 1;
}}
.kpi-sub {{
    font-size: 11px;
    color: {TXT_SEC};
    margin-top: 6px;
}}

.section-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: {TXT_SEC};
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.section-label::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: {BG_BORDER};
}}

.panel {{
    background: {BG_CARD};
    border: 1px solid {BG_BORDER};
    border-radius: 8px;
    padding: 20px;
}}

.badge {{
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
    padding: 3px 10px;
    border-radius: 3px;
    font-weight: 600;
}}

.sidebar-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: {TXT_SEC};
    margin-bottom: 6px;
    margin-top: 16px;
}}
.sidebar-divider {{
    border: none;
    border-top: 1px solid {BG_BORDER};
    margin: 20px 0;
}}
.score-display {{
    background: #0f1520;
    border: 1px solid {BG_BORDER};
    border-radius: 6px;
    padding: 14px 16px;
    text-align: center;
    margin-top: 10px;
}}
.score-num {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 32px;
    font-weight: 600;
    color: {ACCENT};
}}
.score-label {{
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: {TXT_SEC};
    margin-top: 4px;
}}

div[data-testid="stSlider"] > div {{
    padding: 0;
}}
div[data-testid="stSelectbox"] > div > div {{
    background: #0b0e16;
    border-color: {BG_BORDER};
    color: {TXT_PRI};
    border-radius: 6px;
}}
.stRadio > div {{
    gap: 0;
}}
.stRadio label {{
    font-size: 12px;
    padding: 6px 14px;
    border: 1px solid {BG_BORDER};
    margin: 0;
    cursor: pointer;
    color: {TXT_SEC};
    background: #0b0e16;
    transition: all 0.15s;
}}
.stRadio label:first-child {{ border-radius: 4px 0 0 4px; }}
.stRadio label:last-child  {{ border-radius: 0 4px 4px 0; }}
div[data-testid="stDataFrame"] {{
    border: 1px solid {BG_BORDER};
    border-radius: 6px;
    overflow: hidden;
}}
.stTextInput input {{
    background: #0b0e16;
    border: 1px solid {BG_BORDER};
    color: {TXT_PRI};
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
}}
.stDownloadButton button {{
    background: {BG_CARD};
    border: 1px solid {BG_BORDER};
    color: {TXT_PRI};
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 1px;
    border-radius: 4px;
    padding: 8px 18px;
    transition: all 0.15s;
}}
.stDownloadButton button:hover {{
    border-color: {ACCENT};
    color: {ACCENT};
}}

.streamlit-expanderHeader {{
    background: #0b0e16;
    border: 1px solid {BG_BORDER};
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: {TXT_SEC};
}}

button[data-baseweb="tab"] {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: {TXT_SEC};
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {ACCENT};
    border-bottom-color: {ACCENT};
}}
</style>
""", unsafe_allow_html=True)

# ── Data Loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("DataSablon.csv", sep=";")
    df.columns = [c.strip() for c in df.columns]
    df = df[["NAMA PRODUK", "QTY", "HARGA", "TOTAL", "VENDOR", "TANGGAL"]].copy()
    df.dropna(subset=["NAMA PRODUK", "QTY", "HARGA"], inplace=True)

    def clean_rp(s):
        try:
            return float(str(s).replace("Rp", "").replace(".", "").replace(",", "").strip())
        except:
            return np.nan

    df["QTY"]   = pd.to_numeric(df["QTY"], errors="coerce")
    df["HARGA"] = df["HARGA"].apply(clean_rp)
    df["TOTAL"] = df["TOTAL"].apply(clean_rp)
    df.dropna(subset=["QTY", "HARGA"], inplace=True)
    df["QTY"]   = df["QTY"].astype(float)
    df["HARGA"] = df["HARGA"].astype(float)
    return df

df_raw = load_data()
df = df_raw.groupby("NAMA PRODUK").agg(
    QTY=("QTY", "sum"),
    HARGA=("HARGA", "mean"),
    TOTAL=("TOTAL", "sum"),
).reset_index()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
        <div style="padding: 0 4px 20px;">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:18px; font-weight:600;
                        color:{TXT_PRI}; letter-spacing:-1px;">◈ cluster</div>
            <div style="font-size:11px; color:{TXT_SEC}; letter-spacing:2px; margin-top:2px;">ANALYTICS STUDIO</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="sidebar-label">Jumlah Cluster (K)</div>', unsafe_allow_html=True)
    k = st.slider("", min_value=2, max_value=8, value=3, label_visibility="collapsed")

    feature_options = {
        "QTY + HARGA": ["QTY", "HARGA"],
        "QTY + TOTAL": ["QTY", "TOTAL"],
        "HARGA + TOTAL": ["HARGA", "TOTAL"],
        "QTY + HARGA + TOTAL": ["QTY", "HARGA", "TOTAL"],
    }
    st.markdown(f'<div class="sidebar-label">Fitur Analisis</div>', unsafe_allow_html=True)
    fitur_label       = st.selectbox("", list(feature_options.keys()), index=1, label_visibility="collapsed")
    selected_features = feature_options[fitur_label]

    st.markdown(f'<div class="sidebar-label">Mode Tampilan</div>', unsafe_allow_html=True)
    mode = st.radio("", ["Visualisasi", "Evaluasi", "Insight"], label_visibility="collapsed", horizontal=True)

    # ── Run K-Means ──────────────────────────────────────────────────────────
    X        = df[selected_features].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    df["Cluster"] = kmeans.labels_

    centroids_df = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=selected_features,
        index=[f"Cluster {i}" for i in range(k)]
    )

    sil_score = silhouette_score(X_scaled, df["Cluster"]) if k > 1 else 0.0

    inertias = []
    for ki in range(1, 9):
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="score-display">
            <div class="score-num">{sil_score:.2f}</div>
            <div class="score-label">Silhouette Score</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="sidebar-label" style="margin-top:20px;">Elbow Curve</div>', unsafe_allow_html=True)

    fig_elbow, ax_e = plt.subplots(figsize=(3.2, 1.9))
    ax_e.plot(range(1, 9), inertias, color="#6c8ebf", linewidth=2, marker="o",
              markersize=4, markerfacecolor="#0f1117", markeredgecolor="#6c8ebf", markeredgewidth=1.5)
    ax_e.axvline(x=k, color="#d6a35b", linestyle="--", linewidth=1, alpha=0.8)
    ax_e.set_xlabel("K", fontsize=8)
    ax_e.set_ylabel("WCSS", fontsize=7)
    ax_e.tick_params(labelsize=7)
    ax_e.grid(True, alpha=0.2)
    for sp in ax_e.spines.values():
        sp.set_edgecolor(BG_BORDER)
    fig_elbow.tight_layout(pad=0.5)
    st.pyplot(fig_elbow, use_container_width=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="dash-header">
    <div>
        <div class="dash-title">K-Means Clustering — Analisis Penjualan</div>
        <div class="dash-subtitle">{len(df)} produk  ·  {k} cluster  ·  fitur: {" + ".join(selected_features)}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ────────────────────────────────────────────────────────────────────
kpi_html = '<div class="kpi-grid">'
kpi_html += f"""
<div class="kpi-card">
    <div class="kpi-label">Total Produk</div>
    <div class="kpi-value">{len(df)}</div>
    <div class="kpi-sub">SKU unik</div>
</div>
<div class="kpi-card">
    <div class="kpi-label">Silhouette Score</div>
    <div class="kpi-value" style="color:{ACCENT}">{sil_score:.2f}</div>
    <div class="kpi-sub">kualitas cluster</div>
</div>
"""
kpi_html += '</div>'
st.markdown(kpi_html, unsafe_allow_html=True)

# ── Main Grid ──────────────────────────────────────────────────────────────────
cluster_colors = [PALETTE[i % len(PALETTE)] for i in range(k)]
col_l, col_r = st.columns([3, 2], gap="medium")

# ── LEFT: Scatter Plot ─────────────────────────────────────────────────────────
with col_l:
    st.markdown('<div class="section-label">Cluster Scatter</div>', unsafe_allow_html=True)

    feat_x = selected_features[0]
    feat_y = selected_features[1] if len(selected_features) > 1 else selected_features[0]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for ci in range(k):
        mask = df["Cluster"] == ci
        ax.scatter(
            df.loc[mask, feat_x],
            df.loc[mask, feat_y],
            c=cluster_colors[ci], s=35, alpha=0.75, linewidths=0,
            label=f"C{ci}  ({mask.sum()} produk)"
        )

    centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)
    for i, (cx, cy) in enumerate(zip(
        centers_orig[:, selected_features.index(feat_x)],
        centers_orig[:, selected_features.index(feat_y)]
    )):
        ax.scatter(cx, cy, c=cluster_colors[i], marker="D", s=90,
                   edgecolors="white", linewidths=1.5, zorder=6)
        ax.scatter(cx, cy, c="white", marker="+", s=60, linewidths=1.2, zorder=7)

    ax.set_xlabel(feat_x, fontsize=9, labelpad=8)
    ax.set_ylabel(feat_y, fontsize=9, labelpad=8)
    ax.grid(True, alpha=0.15, linestyle="--")
    for sp in ax.spines.values():
        sp.set_edgecolor(BG_BORDER)

    ax.legend(
        loc="upper left", fontsize=9, framealpha=0.0,
        labelcolor=TXT_SEC, handlelength=1.2, handletextpad=0.6
    )
    fig.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)

# ── RIGHT: Mode Panel ──────────────────────────────────────────────────────────
with col_r:
    if mode == "Evaluasi":
        st.markdown('<div class="section-label">Evaluasi Model</div>', unsafe_allow_html=True)

        # ── Elbow ──────────────────────────────────────────────────────────
        st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:10px; letter-spacing:2px; text-transform:uppercase; color:{TXT_SEC}; margin-bottom:6px;">Elbow Method</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5, 3.2))
        ax2.fill_between(range(1, 9), inertias, alpha=0.08, color=ACCENT)
        ax2.plot(range(1, 9), inertias, color=ACCENT, linewidth=2,
                 marker="o", markersize=5, markerfacecolor=BG_DARK,
                 markeredgecolor=ACCENT, markeredgewidth=1.5)
        ax2.axvline(x=k, color="#d6a35b", linestyle="--", linewidth=1.2, alpha=0.9,
                    label=f"K={k} (selected)")
        ax2.set_xlabel("K", fontsize=9)
        ax2.set_ylabel("Inertia (WCSS)", fontsize=9)
        ax2.legend(fontsize=8, framealpha=0, labelcolor=TXT_SEC)
        ax2.grid(True, alpha=0.15, linestyle="--")
        for sp in ax2.spines.values(): sp.set_edgecolor(BG_BORDER)
        fig2.tight_layout(pad=1.2)
        st.pyplot(fig2, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Silhouette ─────────────────────────────────────────────────────
        st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:10px; letter-spacing:2px; text-transform:uppercase; color:{TXT_SEC}; margin-bottom:6px;">Silhouette Score per K</div>', unsafe_allow_html=True)
        sil_vals, k_range = [], range(2, 9)
        for ki in k_range:
            km2 = KMeans(n_clusters=ki, random_state=42, n_init=10)
            sil_vals.append(silhouette_score(X_scaled, km2.fit_predict(X_scaled)))

        best_k = list(k_range)[np.argmax(sil_vals)]
        fig3, ax3 = plt.subplots(figsize=(5, 3.2))
        bar_colors = [ACCENT if ki == k else BG_BORDER for ki in k_range]
        ax3.bar(list(k_range), sil_vals, color=bar_colors, width=0.6,
                edgecolor="none", zorder=3)
        ax3.bar(best_k, sil_vals[best_k - 2], color="#82b366", width=0.6,
                edgecolor="none", zorder=4, label=f"Best K={best_k}")
        ax3.set_xlabel("K", fontsize=9)
        ax3.set_ylabel("Silhouette Score", fontsize=9)
        ax3.legend(fontsize=8, framealpha=0, labelcolor=TXT_SEC)
        ax3.grid(True, axis="y", alpha=0.15, linestyle="--")
        for sp in ax3.spines.values(): sp.set_edgecolor(BG_BORDER)
        fig3.tight_layout(pad=1.2)
        st.pyplot(fig3, use_container_width=True)

    elif mode == "Visualisasi":
        st.markdown('<div class="section-label">Distribusi Cluster</div>', unsafe_allow_html=True)

        summary = df.groupby("Cluster").agg(
            Produk=("NAMA PRODUK", "count"),
            Avg_QTY=("QTY", "mean"),
            Avg_Harga=("HARGA", "mean"),
        ).reset_index()

        fig4, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))

        axes[0].barh(
            [f"C{r}" for r in summary["Cluster"]],
            summary["Avg_QTY"],
            color=[PALETTE[i % len(PALETTE)] for i in summary["Cluster"]],
            height=0.55, edgecolor="none"
        )
        axes[0].set_title("Avg QTY", fontsize=9, pad=8)
        axes[0].grid(True, axis="x", alpha=0.15, linestyle="--")
        axes[0].invert_yaxis()
        for sp in axes[0].spines.values(): sp.set_edgecolor(BG_BORDER)

        axes[1].barh(
            [f"C{r}" for r in summary["Cluster"]],
            summary["Avg_Harga"] / 1000,
            color=[PALETTE[i % len(PALETTE)] for i in summary["Cluster"]],
            height=0.55, edgecolor="none"
        )
        axes[1].set_title("Avg Harga (ribu Rp)", fontsize=9, pad=8)
        axes[1].grid(True, axis="x", alpha=0.15, linestyle="--")
        axes[1].invert_yaxis()
        for sp in axes[1].spines.values(): sp.set_edgecolor(BG_BORDER)

        fig4.tight_layout(pad=1.2)
        st.pyplot(fig4, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Ukuran Cluster</div>', unsafe_allow_html=True)
        cols_badge = st.columns(k)
        for ci in range(k):
            n   = (df["Cluster"] == ci).sum()
            pct = n / len(df) * 100
            with cols_badge[ci]:
                st.markdown(f"""
                    <div style="background:{BG_CARD}; border:1px solid {BG_BORDER};
                                border-top: 2px solid {PALETTE[ci % len(PALETTE)]};
                                border-radius:6px; padding:12px 14px; text-align:center;">
                        <div style="font-family:'IBM Plex Mono',monospace; font-size:22px;
                                    font-weight:600; color:{PALETTE[ci % len(PALETTE)]};">{n}</div>
                        <div style="font-size:10px; color:{TXT_SEC}; margin-top:4px;">
                            C{ci} · {pct:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)

    elif mode == "Insight":
        st.markdown('<div class="section-label">Insight per Cluster</div>', unsafe_allow_html=True)

        for ci in range(k):
            grp   = df[df["Cluster"] == ci]
            color = PALETTE[ci % len(PALETTE)]

            with st.expander(f"Cluster {ci}  —  {len(grp)} produk", expanded=(ci == 0)):
                top3 = grp.nlargest(3, "QTY")[["NAMA PRODUK", "QTY", "HARGA"]]
                top3["HARGA"] = top3["HARGA"].apply(lambda x: f"Rp {x:,.0f}")
                top3["QTY"]   = top3["QTY"].astype(int)
                st.caption("Top 3 Produk berdasarkan QTY")
                st.dataframe(
                    top3.rename(columns={"NAMA PRODUK": "Produk"}),
                    use_container_width=True, hide_index=True
                )

                # ── Rekomendasi Strategi per Cluster ─────────────────────────
                avg_qty_all   = df["QTY"].mean()
                avg_total_all = df["TOTAL"].mean()

                cluster_avg_qty   = grp["QTY"].mean()
                cluster_avg_total = grp["TOTAL"].mean()

                if cluster_avg_qty >= avg_qty_all and cluster_avg_total >= avg_total_all:
                    kategori = "🔥 Produk Sangat Laku"
                    rekomendasi = [
                        "Pastikan stok selalu tersedia agar tidak kehabisan.",
                        "Prioritaskan produk ini dalam promosi dan iklan.",
                        "Gunakan strategi bundling atau upselling untuk meningkatkan omzet."
                    ]
                elif cluster_avg_qty >= avg_qty_all:
                    kategori = "📈 Produk Potensial"
                    rekomendasi = [
                        "Tingkatkan visibilitas produk melalui promosi khusus.",
                        "Evaluasi harga untuk memaksimalkan margin keuntungan.",
                        "Jadikan produk ini sebagai produk unggulan untuk menarik pelanggan baru."
                    ]
                else:
                    kategori = "⚠️ Produk Kurang Laku"
                    rekomendasi = [
                        "Berikan diskon atau paket bundling untuk meningkatkan penjualan.",
                        "Evaluasi harga, desain, atau kualitas produk.",
                        "Kurangi stok dan alihkan fokus ke produk yang lebih diminati pasar."
                    ]

                st.markdown(f"### {kategori}")
                for rec in rekomendasi:
                    st.markdown(f"- {rec}")

# ── Data Table ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Data Hasil Clustering</div>', unsafe_allow_html=True)

col_f1, col_f2, col_f3 = st.columns([3, 2, 1])
with col_f1:
    search = st.text_input("Cari produk…", "", label_visibility="collapsed",
                           placeholder="Cari produk…")
with col_f2:
    cluster_filter = st.selectbox(
        "Filter", ["Semua Cluster"] + [f"Cluster {i}" for i in range(k)],
        label_visibility="collapsed"
    )
with col_f3:
    show_all = st.checkbox("Semua", value=False)

display_df = df[["NAMA PRODUK", "QTY", "HARGA", "TOTAL", "Cluster"]].copy()
display_df["QTY"]     = display_df["QTY"].astype(int)
display_df["HARGA"]   = display_df["HARGA"].apply(lambda x: f"Rp {x:,.0f}")
display_df["TOTAL"]   = display_df["TOTAL"].apply(lambda x: f"Rp {x:,.0f}")
display_df["Cluster"] = display_df["Cluster"].apply(lambda c: f"Cluster {c}")
display_df.columns    = ["Nama Produk", "Qty", "Harga", "Total", "Cluster"]

filtered = display_df.copy()
if search:
    filtered = filtered[filtered["Nama Produk"].str.contains(search, case=False)]
if cluster_filter != "Semua Cluster":
    filtered = filtered[filtered["Cluster"] == cluster_filter]

n_show = len(filtered) if show_all else min(50, len(filtered))
st.dataframe(filtered.head(n_show), use_container_width=True, hide_index=True, height=300)
st.caption(f"{n_show} dari {len(filtered)} data ditampilkan")

# ── Download ───────────────────────────────────────────────────────────────────
out_df = df[["NAMA PRODUK", "QTY", "HARGA", "TOTAL", "Cluster"]].copy()
out_df["Cluster"] = out_df["Cluster"].apply(lambda c: f"Cluster {c}")
st.download_button(
    "↓  Export CSV",
    out_df.to_csv(index=False).encode("utf-8"),
    "hasil_clustering.csv", "text/csv"
)