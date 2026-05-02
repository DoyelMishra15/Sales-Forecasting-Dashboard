import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Intelligence Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL STYLES ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=Inter:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #12121f 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.15);
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'Syne', sans-serif;
    color: #a5b4fc;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
}

/* Hero title */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc 0%, #818cf8 40%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #6366f1;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* KPI cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.kpi-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(168,85,247,0.05) 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s ease;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6366f1, #a855f7);
}
.kpi-card:hover { border-color: rgba(99,102,241,0.5); }
.kpi-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.6rem;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #e8e8f0;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.kpi-delta {
    font-size: 0.8rem;
    font-weight: 500;
    color: #34d399;
}
.kpi-delta.neg { color: #f87171; }

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #c7d2fe;
    letter-spacing: 0.02em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #6366f1;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 4px;
    padding: 2px 8px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Insight cards */
.insight-card {
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    font-size: 0.88rem;
    line-height: 1.6;
    color: #c7d2fe;
}
.insight-card strong { color: #a5b4fc; }

/* Data table */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(99,102,241,0.15) !important;
}

/* Sliders & inputs */
.stSlider [data-baseweb="slider"] {
    padding-top: 0.5rem;
}
.stSlider [data-testid="stMarkdownContainer"] p {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #818cf8 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s ease;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(99,102,241,0.35);
}

/* Divider */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent);
    margin: 2rem 0;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(99,102,241,0.05);
    border: 1.5px dashed rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 1rem;
}

/* Metric */
[data-testid="stMetric"] {
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 12px;
    padding: 1rem;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(99,102,241,0.06);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(99,102,241,0.15);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #818cf8;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.05em;
    padding: 0.5rem 1.2rem;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ─── MATPLOTLIB STYLE ──────────────────────────────────────────────────────────
def apply_dark_style(fig, ax_list=None):
    fig.patch.set_facecolor('#0f0f1a')
    axes = ax_list if ax_list is not None else fig.get_axes()
    if hasattr(axes, 'flatten'):  # handle numpy 2D arrays from subplots
        axes = axes.flatten().tolist()
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor('#13131f')
        ax.tick_params(colors='#6b7280', labelsize=9)
        ax.xaxis.label.set_color('#9ca3af')
        ax.yaxis.label.set_color('#9ca3af')
        ax.title.set_color('#c7d2fe')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1f2035')
        ax.grid(color='#1f2035', linewidth=0.7, linestyle='--', alpha=0.8)
    return fig


# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-family: Syne, sans-serif; font-size:1.5rem; font-weight:800;
                    background: linear-gradient(135deg,#a5b4fc,#c084fc);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            SALES IQ
        </div>
        <div style='font-family: DM Mono, monospace; font-size:0.65rem;
                    color:#6366f1; letter-spacing:0.2em; text-transform:uppercase;'>
            Intelligence Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## ⚙ CONFIGURATION")

    uploaded_file = st.file_uploader(
        "Upload CSV (Date + Sales)",
        type=["csv"],
        help="CSV with 'Date' (YYYY-MM-DD) and 'Sales' columns"
    )

    st.markdown("---")
    st.markdown("## 🔮 FORECAST")
    period = st.slider("Forecast horizon (days)", 30, 365, 90, 30)
    growth_mode = st.selectbox("Growth model", ["linear", "logistic"], index=0)

    st.markdown("## 📊 ANALYSIS")
    show_components = st.checkbox("Show seasonality components", True)
    show_regression = st.checkbox("Show regression analysis", True)
    poly_degree = st.slider("Polynomial degree", 1, 4, 1)

    st.markdown("---")
    st.markdown("""
    <div style='font-family: DM Mono, monospace; font-size:0.7rem; color:#4b5563;
                text-align:center; padding: 0.5rem;'>
        v2.0 · Prophet + Sklearn<br>Built with Streamlit
    </div>
    """, unsafe_allow_html=True)


# ─── MAIN AREA ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-title">Sales Intelligence</div>
<div class="hero-sub">▸ Forecasting Dashboard · Prophet · ML Regression</div>
""", unsafe_allow_html=True)

if not uploaded_file:
    # ── LANDING STATE ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(168,85,247,0.06));
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
    '>
        <div style='font-size:3.5rem; margin-bottom:1rem;'>📂</div>
        <div style='font-family: Syne, sans-serif; font-size:1.4rem; font-weight:700;
                    color:#c7d2fe; margin-bottom:0.5rem;'>
            Upload your sales data to begin
        </div>
        <div style='font-family: DM Mono, monospace; font-size:0.82rem; color:#6366f1;'>
            CSV format · Date column (YYYY-MM-DD) · Sales column (numeric)
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    features = [
        ("🔮", "Prophet Forecasting", "Facebook's battle-tested time-series model for accurate multi-period predictions"),
        ("📉", "Regression Analysis", "Linear & polynomial trend fitting with R² score, MSE, and RMSE metrics"),
        ("📊", "Interactive Insights", "Automated business insights extracted from your data patterns"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], features):
        col.markdown(f"""
        <div class='kpi-card' style='text-align:center; padding:1.8rem 1.2rem;'>
            <div style='font-size:2rem; margin-bottom:0.8rem;'>{icon}</div>
            <div style='font-family:Syne,sans-serif; font-weight:700; color:#c7d2fe;
                        font-size:0.95rem; margin-bottom:0.5rem;'>{title}</div>
            <div style='font-size:0.8rem; color:#6b7280; line-height:1.5;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

else:
    # ── LOAD DATA ──────────────────────────────────────────────────────────────
    try:
        df_raw = pd.read_csv(uploaded_file)
        df = df_raw.rename(columns={'Date': 'ds', 'Sales': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').reset_index(drop=True)
    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
        st.stop()

    # ── KPI METRICS ────────────────────────────────────────────────────────────
    total_sales   = df['y'].sum()
    avg_daily     = df['y'].mean()
    peak_sales    = df['y'].max()
    peak_date     = df.loc[df['y'].idxmax(), 'ds'].strftime('%b %d, %Y')
    n_days        = (df['ds'].max() - df['ds'].min()).days
    recent_avg    = df.tail(30)['y'].mean()
    earlier_avg   = df.head(30)['y'].mean()
    mom_delta     = ((recent_avg - earlier_avg) / earlier_avg) * 100 if earlier_avg else 0

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Total Revenue</div>
            <div class="kpi-value">{total_sales:,.0f}</div>
            <div class="kpi-delta">↑ {n_days} days tracked</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Daily Average</div>
            <div class="kpi-value">{avg_daily:,.0f}</div>
            <div class="kpi-delta {'neg' if mom_delta < 0 else ''}">
                {'↑' if mom_delta >= 0 else '↓'} {abs(mom_delta):.1f}% vs first 30d
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Peak Day</div>
            <div class="kpi-value">{peak_sales:,.0f}</div>
            <div class="kpi-delta">📅 {peak_date}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Recent 30d Avg</div>
            <div class="kpi-value">{recent_avg:,.0f}</div>
            <div class="kpi-delta {'neg' if mom_delta < 0 else ''}">
                {'↑' if mom_delta >= 0 else '↓'} Momentum
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TRAIN PROPHET (shared across all tabs) ─────────────────────────────────
    with st.spinner("Training Prophet model..."):
        df_prophet = df[['ds', 'y']].copy()
        if growth_mode == 'logistic':
            df_prophet['cap'] = df_prophet['y'].max() * 1.5
            df_prophet['floor'] = 0
        model = Prophet(growth=growth_mode, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=period)
        if growth_mode == 'logistic':
            future['cap'] = df_prophet['cap'].max()
            future['floor'] = 0
        forecast = model.predict(future)

    # ── TABS ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 FORECAST",
        "📊 REGRESSION",
        "📋 DATA EXPLORER",
        "🧠 INSIGHTS",
    ])

    # ── TAB 1: PROPHET FORECAST ─────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">Prophet Forecast <span class="section-tag">ML</span></div>', unsafe_allow_html=True)

        # Custom forecast plot
        fig, ax = plt.subplots(figsize=(12, 5))
        apply_dark_style(fig, [ax])

        hist = forecast[forecast['ds'] <= df['ds'].max()]
        pred = forecast[forecast['ds'] > df['ds'].max()]

        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                        color='#6366f1', alpha=0.12, label='Confidence Interval')
        ax.plot(hist['ds'], hist['yhat'], color='#818cf8', linewidth=1.5, alpha=0.7, label='Fitted')
        ax.plot(pred['ds'], pred['yhat'], color='#a855f7', linewidth=2.2, linestyle='--', label=f'Forecast (+{period}d)')
        ax.scatter(df['ds'], df['y'], color='#c7d2fe', s=12, alpha=0.6, zorder=5, label='Actual')

        ax.axvline(df['ds'].max(), color='#6366f1', linewidth=1, linestyle=':', alpha=0.8)
        ax.set_title(f'Sales Forecast — Next {period} Days', fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Date'), ax.set_ylabel('Sales')
        ax.legend(loc='upper left', fontsize=8, facecolor='#13131f', edgecolor='#1f2035', labelcolor='#9ca3af')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Forecast summary
        future_only = forecast[forecast['ds'] > df['ds'].max()]
        col1, col2, col3 = st.columns(3)
        col1.metric("Forecasted Total", f"{future_only['yhat'].sum():,.0f}", f"+{period} days")
        col2.metric("Forecasted Avg/Day", f"{future_only['yhat'].mean():,.1f}")
        col3.metric("Forecasted Peak", f"{future_only['yhat'].max():,.0f}")

        if show_components:
            st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Seasonality Components</div>', unsafe_allow_html=True)
            fig2 = model.plot_components(forecast)
            apply_dark_style(fig2, fig2.get_axes())
            fig2.patch.set_facecolor('#0f0f1a')
            fig2.set_size_inches(12, 7)
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close()

    # ── TAB 2: REGRESSION ──────────────────────────────────────────────────
    with tab2:
        if show_regression:
            st.markdown('<div class="section-header">Trend Analysis <span class="section-tag">Regression</span></div>', unsafe_allow_html=True)

            df_reg = df.copy()
            df_reg['timestamp'] = df_reg['ds'].map(pd.Timestamp.toordinal)
            X = df_reg[['timestamp']]
            y_vals = df_reg['y']

            pipeline = make_pipeline(PolynomialFeatures(poly_degree), LinearRegression())
            pipeline.fit(X, y_vals)
            y_pred = pipeline.predict(X)

            mse  = mean_squared_error(y_vals, y_pred)
            rmse = np.sqrt(mse)
            r2   = r2_score(y_vals, y_pred)
            residuals = y_vals - y_pred

            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            apply_dark_style(fig, axes)

            # Plot 1 – trend
            axes[0].fill_between(df_reg['ds'], y_pred - rmse, y_pred + rmse,
                                  color='#f59e0b', alpha=0.1, label='±1 RMSE band')
            axes[0].plot(df_reg['ds'], df_reg['y'], color='#818cf8', linewidth=1.2, alpha=0.8, label='Actual Sales')
            axes[0].plot(df_reg['ds'], y_pred, color='#f59e0b', linewidth=2.2, label=f'Poly-{poly_degree} Trend')
            axes[0].set_title(f'Sales Trend (Degree {poly_degree})', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Date'), axes[0].set_ylabel('Sales')
            axes[0].legend(fontsize=8, facecolor='#13131f', edgecolor='#1f2035', labelcolor='#9ca3af')

            # Plot 2 – residuals
            axes[1].bar(df_reg['ds'], residuals, color=np.where(residuals >= 0, '#34d399', '#f87171'), alpha=0.7, width=1)
            axes[1].axhline(0, color='#6366f1', linewidth=1)
            axes[1].set_title('Residuals (Actual − Predicted)', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Date'), axes[1].set_ylabel('Residual')

            fig.tight_layout(pad=2)
            st.pyplot(fig)
            plt.close()

            col1, col2, col3 = st.columns(3)
            col1.metric("R² Score", f"{r2:.4f}", "Model fit quality")
            col2.metric("RMSE", f"{rmse:,.2f}", "Root Mean Squared Error")
            col3.metric("MSE", f"{mse:,.2f}", "Mean Squared Error")

    # ── TAB 3: DATA EXPLORER ───────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">Data Explorer <span class="section-tag">CSV</span></div>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            n = st.slider("Rows to preview", 5, len(df), min(20, len(df)), 5)
            st.dataframe(
                df.rename(columns={'ds': 'Date', 'y': 'Sales'}).tail(n),
                use_container_width=True,
                hide_index=True
            )
        with col2:
            st.markdown("**Distribution**")
            fig, ax = plt.subplots(figsize=(5, 4))
            apply_dark_style(fig, [ax])
            ax.hist(df['y'], bins=25, color='#6366f1', alpha=0.8, edgecolor='#0f0f1a', linewidth=0.5)
            ax.axvline(df['y'].mean(), color='#f59e0b', linewidth=1.8, linestyle='--', label='Mean')
            ax.axvline(df['y'].median(), color='#34d399', linewidth=1.8, linestyle=':', label='Median')
            ax.legend(fontsize=8, facecolor='#13131f', edgecolor='#1f2035', labelcolor='#9ca3af')
            ax.set_title('Sales Distribution', fontsize=10)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("**Rolling Statistics**")
        fig, ax = plt.subplots(figsize=(12, 3.5))
        apply_dark_style(fig, [ax])
        ax.plot(df['ds'], df['y'], color='#818cf8', linewidth=1, alpha=0.5, label='Daily')
        ax.plot(df['ds'], df['y'].rolling(7).mean(), color='#a855f7', linewidth=2, label='7-day MA')
        ax.plot(df['ds'], df['y'].rolling(30).mean(), color='#f59e0b', linewidth=2, label='30-day MA')
        ax.set_title('Moving Averages', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, facecolor='#13131f', edgecolor='#1f2035', labelcolor='#9ca3af')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── TAB 4: INSIGHTS ────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">Automated Insights <span class="section-tag">AI</span></div>', unsafe_allow_html=True)

        # Day-of-week analysis
        df['dow'] = df['ds'].dt.day_name()
        dow_avg = df.groupby('dow')['y'].mean().reindex(
            ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        )
        best_day  = dow_avg.idxmax()
        worst_day = dow_avg.idxmin()

        # Monthly analysis
        df['month'] = df['ds'].dt.month_name()
        monthly = df.groupby('month')['y'].mean()
        best_month = monthly.idxmax() if len(monthly) > 1 else "N/A"

        insights = [
            (f"📅 <strong>{best_day}</strong> is your highest-performing day of the week "
             f"with an average of <strong>{dow_avg[best_day]:,.0f}</strong> in sales."),
            (f"📉 <strong>{worst_day}</strong> is your lowest-performing day "
             f"(avg <strong>{dow_avg[worst_day]:,.0f}</strong>). Consider promotions on this day."),
            (f"🏆 Your all-time peak was <strong>{peak_sales:,.0f}</strong> units on "
             f"<strong>{peak_date}</strong>."),
            (f"📈 Recent 30-day momentum is "
             f"<strong>{'positive ↑' if mom_delta >= 0 else 'negative ↓'} {abs(mom_delta):.1f}%</strong> "
             f"compared to the first 30 days of data."),
            (f"🔮 The Prophet model forecasts an average of "
             f"<strong>{forecast[forecast['ds'] > df['ds'].max()]['yhat'].mean():,.1f}</strong> "
             f"daily sales over the next <strong>{period} days</strong>."),
        ]

        for ins in insights:
            st.markdown(f'<div class="insight-card">{ins}</div>', unsafe_allow_html=True)

        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Day-of-Week Performance</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(9, 3.5))
        apply_dark_style(fig, [ax])
        colors = ['#a855f7' if d == best_day else '#6366f1' for d in dow_avg.index]
        bars = ax.bar(dow_avg.index, dow_avg.values, color=colors, alpha=0.85, width=0.65, edgecolor='#0f0f1a')
        for bar, val in zip(bars, dow_avg.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:,.0f}', ha='center', va='bottom', fontsize=8, color='#9ca3af',
                    fontfamily='monospace')
        ax.set_title('Average Sales by Day of Week', fontsize=11, fontweight='bold')
        ax.set_ylabel('Avg Sales')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()