import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="Live Cattle Price Predictor", page_icon="🐄", layout="wide")

st.title("🐄 CME Live Cattle Futures — 1-Month Price Predictor")
st.markdown("*Predict next month's Live Cattle (LE) futures price using USDA supply data and market inputs*")
st.markdown("---")

@st.cache_resource
def load_artifacts():
    model           = joblib.load("cattle_price_model.pkl")
    scaler          = joblib.load("scaler.pkl")
    feature_cols    = joblib.load("feature_cols.pkl")
    feature_medians = joblib.load("feature_medians.pkl")
    return model, scaler, feature_cols, feature_medians

@st.cache_data(ttl=3600)
def fetch_live_prices():
    """Fetch last 4 monthly closes for LE=F, GF=F, ZC=F from yfinance."""
    defaults = {"le": [248.25, 258.20, 226.68, 220.35], "gf": [295.0, 290.0], "corn": [480.0, 475.0]}
    try:
        le   = yf.download("LE=F", period="12mo", interval="1mo", progress=False, auto_adjust=True)["Close"].dropna()
        gf   = yf.download("GF=F", period="12mo", interval="1mo", progress=False, auto_adjust=True)["Close"].dropna()
        corn = yf.download("ZC=F", period="12mo", interval="1mo", progress=False, auto_adjust=True)["Close"].dropna()

        def last_n(series, n):
            vals = series.squeeze().tolist()
            if len(vals) == 0:
                return None
            # pad with first value if not enough history
            while len(vals) < n:
                vals.insert(0, vals[0])
            return [round(v, 2) for v in vals[-n:]]

        le_vals   = last_n(le,   4)
        gf_vals   = last_n(gf,   2)
        corn_vals = last_n(corn, 2)

        return {
            "le":   le_vals   or defaults["le"],
            "gf":   gf_vals   or defaults["gf"],
            "corn": corn_vals or defaults["corn"],
            "live": True,
        }
    except Exception:
        return {**defaults, "live": False}

try:
    model, scaler, feature_cols, feature_medians = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}. Run the notebook first to generate .pkl files.")
    st.stop()

prices = fetch_live_prices()
le_d   = prices["le"]
gf_d   = prices["gf"]
corn_d = prices["corn"]

if not prices.get("live"):
    st.warning("⚠️ Could not fetch live prices — showing recent defaults. Update sidebar inputs manually.")

# ── Sidebar inputs ─────────────────────────────────────────────────────────────
st.sidebar.header("Market Inputs")
st.sidebar.caption("Defaults auto-populated from live CME prices via yfinance")

st.sidebar.subheader("📅 Forecast Month")
pred_month = st.sidebar.selectbox(
    "Next Month to Predict",
    range(1, 13),
    index=datetime.now().month % 12,
    format_func=lambda x: datetime(2000, x, 1).strftime("%B")
)

st.sidebar.subheader("💰 Recent Prices")
le_1   = st.sidebar.number_input("Live Cattle — Last Month (¢/lb)",        80.0, 320.0, float(le_d[-1]),   step=0.5)
le_2   = st.sidebar.number_input("Live Cattle — 2 Months Ago (¢/lb)",      80.0, 320.0, float(le_d[-2]),   step=0.5)
le_3   = st.sidebar.number_input("Live Cattle — 3 Months Ago (¢/lb)",      80.0, 320.0, float(le_d[-3]),   step=0.5)
le_12m = st.sidebar.number_input("Live Cattle — 12-Mo Avg (¢/lb)",         80.0, 320.0, float(le_d[-4]),   step=0.5)
gf_1   = st.sidebar.number_input("Feeder Cattle — Last Month (¢/lb)",      90.0, 420.0, float(gf_d[-1]),   step=0.5)
gf_2   = st.sidebar.number_input("Feeder Cattle — 2 Months Ago (¢/lb)",    90.0, 420.0, float(gf_d[-2]),   step=0.5)
corn_1 = st.sidebar.number_input("Corn — Last Month (¢/bu)",               150.0, 900.0, float(corn_d[-1]), step=1.0)
corn_2 = st.sidebar.number_input("Corn — 2 Months Ago (¢/bu)",             150.0, 900.0, float(corn_d[-2]), step=1.0)

st.sidebar.subheader("🐮 Supply (Last Month)")
beef_prod  = st.sidebar.number_input("Beef Production (mil lbs)",           1500.0, 2800.0, 2200.0, step=10.0)
beef_yoy   = st.sidebar.number_input("Beef Prod vs Year Ago (%)",            -20.0,   20.0,    0.0, step=0.1)
slaughter  = st.sidebar.number_input("Cattle Slaughter (1,000 head)",       1500.0, 3500.0, 2600.0, step=10.0)
cold_stor  = st.sidebar.number_input("Beef Cold Storage (mil lbs)",          200.0,  700.0,  420.0, step=5.0)
cattle_inv = st.sidebar.number_input("Cattle Inventory (mil head)",           80.0,  120.0,   92.0, step=0.5)

st.sidebar.subheader("🌍 Market Context")
dxy = st.sidebar.number_input("US Dollar Index (DXY)", 70.0, 120.0, 101.0, step=0.1)

# ── Feature construction ───────────────────────────────────────────────────────
def build_feature_row():
    vals = {
        "live_cattle_price_lag1":      le_1,
        "live_cattle_price_lag2":      le_2,
        "live_cattle_price_lag3":      le_3,
        "live_cattle_price_roll3":     np.mean([le_1, le_2, le_3]),
        "live_cattle_price_roll6":     np.mean([le_1, le_2, le_3, le_12m, le_12m, le_12m]),
        "live_cattle_price_roll12":    le_12m,
        "feeder_cattle_price_lag1":    gf_1,
        "feeder_cattle_price_lag2":    gf_2,
        "feeder_cattle_price_roll3":   np.mean([gf_1, gf_2, gf_2]),
        "corn_price_lag1":             corn_1,
        "corn_price_lag2":             corn_2,
        "corn_price_roll3":            np.mean([corn_1, corn_2, corn_2]),
        "comm_beef_prod_lag1":         beef_prod,
        "comm_beef_prod_yoy":          beef_yoy / 100.0,
        "comm_cattle_slaughter_lag1":  slaughter,
        "beef_cold_storage_lag1":      cold_stor,
        "cattle_inventory_lag1":       cattle_inv * 1_000_000,
        "cattle_inventory_yoy":        0.0,
        "dxy_lag1":                    dxy,
        "dxy_roll3":                   dxy,
        "feed_cost_ratio":             corn_1 / le_1 if le_1 > 0 else 1.0,
        "prod_cycle":                  beef_prod,
    }
    for m in range(2, 13):
        vals[f"month_{m}"] = 1 if m == pred_month else 0

    row = {col: vals.get(col, feature_medians.get(col, 0.0)) for col in feature_cols}
    return pd.DataFrame([row])

# ── Prediction ─────────────────────────────────────────────────────────────────
X_input    = build_feature_row()
X_scaled   = scaler.transform(X_input)
prediction = model.predict(X_scaled)[0]
delta      = prediction - le_1
pct_change = delta / le_1 * 100

st.header("Forecast Result")
c1, c2, c3 = st.columns(3)
c1.metric("Current LE Price",          f"{le_1:.2f} ¢/lb")
c2.metric("Predicted LE (Next Month)", f"{prediction:.2f} ¢/lb", f"{delta:+.2f} ¢/lb")
c3.metric("Expected Move",             f"{pct_change:+.1f}%",
          delta="Bullish" if pct_change > 0 else "Bearish",
          delta_color="normal" if pct_change > 0 else "inverse")

st.markdown("---")

# ── Input summary ──────────────────────────────────────────────────────────────
st.subheader("Inputs Summary")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
| Input | Value |
|-------|-------|
| Live Cattle — Current Price | {le_1:.2f} ¢/lb |
| Live Cattle — 2 Months Ago | {le_2:.2f} ¢/lb |
| Live Cattle — 12-Mo Avg | {le_12m:.2f} ¢/lb |
| Feeder Cattle — Current Price | {gf_1:.2f} ¢/lb |
| Corn — Current Price | {corn_1:.1f} ¢/bu |
| Feed Cost Ratio | {corn_1/le_1:.3f} |
""")

with col2:
    st.markdown(f"""
| Input | Value |
|-------|-------|
| Beef Production | {beef_prod:,.0f} mil lbs |
| YoY Change | {beef_yoy:+.1f}% |
| Cold Storage | {cold_stor:,.0f} mil lbs |
| Cattle Inventory | {cattle_inv:.1f} mil head |
| Dollar Index | {dxy:.2f} |
| Forecast Month | {datetime(2000, pred_month, 1).strftime("%B")} |
""")

st.markdown("---")
st.caption("Prices auto-fetched from CME via yfinance · Refreshed hourly · Model: Ridge/Lasso Regression | Data: USDA ERS + CME Futures | CRISP-DM Capstone — Brian Place")
