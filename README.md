# 📈 Sales Intelligence Dashboard

> A premium, production-grade sales forecasting dashboard built with **Streamlit**, **Facebook Prophet**, and **Scikit-learn** — featuring a stunning dark UI, automated insights, and multi-model analytics.

---

## ✨ What's New in v2.0

| Feature | v1 | v2 |
|---|---|---|
| UI Theme | Basic white | Dark glassmorphism |
| KPI Cards | ❌ | ✅ 4 animated metric cards |
| Tabs | ❌ | ✅ Forecast · Regression · Explorer · Insights |
| Regression | Linear only | Linear + Polynomial (degree 1–4) |
| Charts | Default matplotlib | Custom dark-themed charts |
| Moving Averages | ❌ | ✅ 7d & 30d rolling MA |
| Residual Plot | ❌ | ✅ Color-coded residuals |
| Day-of-week analysis | ❌ | ✅ Automated insights |
| Distribution histogram | ❌ | ✅ With mean/median markers |
| Logistic growth model | ❌ | ✅ Prophet logistic mode |
| Google Fonts | ❌ | ✅ Syne + DM Mono + Inter |

---

## 🚀 Features

- **Upload CSV** with `Date` and `Sales` columns
- **KPI Dashboard** — Total revenue, daily average, peak sales, momentum
- **🔮 Forecast Tab** — Prophet forecast with confidence intervals, horizon selector, growth modes
- **📊 Regression Tab** — Polynomial regression (degree 1–4), residual chart, R², RMSE, MSE
- **📋 Data Explorer** — Rolling averages (7d/30d), distribution histogram, configurable preview
- **🧠 Insights Tab** — Automated business insights: best/worst days, momentum, peak detection, forecast summary

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Streamlit** — Dashboard framework
- **Facebook Prophet** — Time-series forecasting
- **Scikit-learn** — Polynomial regression, metrics
- **Pandas / NumPy** — Data processing
- **Matplotlib** — Custom dark-themed charts

---

## ⚡ Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/DoyelMishra15/Sales-Forecasting-Dashboard.git
cd Sales-Forecasting-Dashboard/sales_dashboard

# 2. Install dependencies
pip install streamlit prophet scikit-learn pandas matplotlib numpy

# 3. Run the app
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) and upload the included `sample_sales_data.csv`.

---

## 📁 CSV Format

```
Date,Sales
2023-01-01,195
2023-01-02,160
...
```

---

## 📸 Dashboard Sections

```
┌─────────────────────────────────────────────────┐
│  SIDEBAR          │  MAIN CONTENT               │
│  ─────────────    │  ─────────────────────────  │
│  Upload CSV       │  KPI Cards (4 metrics)      │
│  Forecast days    │                             │
│  Growth model     │  ┌─────────────────────┐   │
│  Components ✓     │  │ 🔮 Forecast          │   │
│  Regression ✓     │  │ 📊 Regression        │   │
│  Poly degree      │  │ 📋 Data Explorer     │   │
│                   │  │ 🧠 Insights          │   │
│                   │  └─────────────────────┘   │
└─────────────────────────────────────────────────┘
```

---

## 👤 Author

**Doyel Mishra** — [GitHub @DoyelMishra15](https://github.com/DoyelMishra15)
