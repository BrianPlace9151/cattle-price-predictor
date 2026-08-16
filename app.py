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
    defaults = {
        "le": [248.25, 258.20, 226.68],
        "gf": [295.0, 290.0, 285.0],
        "corn_roll6": 424.83,
        "corn_roll12": 437.25,
        "soymeal_roll3": 340.97,
        "month_open": 243.0,
        "live": False,
    }
    try:
        def get_monthly(ticker, n):
            df = yf.Ticker(ticker).history(period="14mo", interval="1mo")
            close = df["Close"].dropna()
            vals  = close.tolist()
            while len(vals) < n:
                vals.insert(0, vals[0])
            return [round(v, 2) for v in vals[-n:]], df

        le_vals,   le_df   = get_monthly("LE=F", 3)
        gf_vals,   _       = get_monthly("GF=F", 3)
        corn_vals, _       = get_monthly("ZC=F", 12)
        soy_vals,  _       = get_monthly("ZM=F", 3)

        month_open = round(float(le_df["Open"].iloc[-1]), 2) if not le_df.empty else defaults["month_open"]

        return {
            "le": le_vals,
            "gf": gf_vals,
            "corn_roll6":    round(float(np.mean(corn_vals[-6:])), 2),
            "corn_roll12":   round(float(np.mean(corn_vals[-12:])), 2),
            "soymeal_roll3": round(float(np.mean(soy_vals[-3:])), 2),
            "month_open": month_open,
            "live": True,
        }
    except Exception:
        return defaults

@st.cache_data(ttl=3600)
def fetch_local_market_data():
    """Managed-money positioning and packer-margin data — same source files the
    notebook trains on (CFTC COT report, USDA AMS Choice/Select cutout), not
    available via yfinance."""
    defaults = {"mm_net": 65208.0, "choice_cutout": 215.39, "cutout_spread_lag1": 10.35, "live": False}
    try:
        cot = pd.read_csv("cftc_cot_cattle.csv", low_memory=False)
        cot.columns = cot.columns.str.strip()
        lc = cot[cot["Market_and_Exchange_Names"].str.upper().str.contains("LIVE CATTLE", na=False)].copy()
        lc["date"] = pd.to_datetime(lc["Report_Date_as_YYYY-MM-DD"], errors="coerce")
        lc = lc.dropna(subset=["date"]).set_index("date").sort_index()
        lc["mm_net"] = (pd.to_numeric(lc["M_Money_Positions_Long_All"],  errors="coerce") -
                         pd.to_numeric(lc["M_Money_Positions_Short_All"], errors="coerce"))
        mm_net_series = lc["mm_net"].resample("ME").last().dropna()
        mm_net = round(float(mm_net_series.iloc[-1]), 0)

        cutout = pd.read_csv("choice_cutout.csv")
        cutout["date"]           = pd.to_datetime(cutout["Report Date"], errors="coerce")
        cutout["choice_cutout"]  = pd.to_numeric(cutout["Choice 600-900"], errors="coerce")
        cutout["select_cutout"]  = pd.to_numeric(cutout["Select 600-900"], errors="coerce")
        cutout = (cutout.dropna(subset=["date", "choice_cutout", "select_cutout"])
                        .set_index("date").sort_index())
        cutout_m = cutout[["choice_cutout", "select_cutout"]].resample("ME").last()
        cutout_m["cutout_spread"] = cutout_m["choice_cutout"] - cutout_m["select_cutout"]

        choice_cutout      = round(float(cutout_m["choice_cutout"].iloc[-1]), 2)
        cutout_spread_lag1 = round(float(cutout_m["cutout_spread"].iloc[-2]), 2)

        return {"mm_net": mm_net, "choice_cutout": choice_cutout,
                "cutout_spread_lag1": cutout_spread_lag1, "live": True}
    except Exception:
        return defaults

try:
    model, scaler, feature_cols, feature_medians = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}. Run the notebook first to generate .pkl files.")
    st.stop()

prices      = fetch_live_prices()
local_data  = fetch_local_market_data()
le_d        = prices["le"]
gf_d        = prices["gf"]
month_open  = prices.get("month_open")

if not prices.get("live") or not local_data.get("live"):
    st.warning("⚠️ Could not fetch some live data — showing recent defaults. Update sidebar inputs manually.")

# ── Sidebar inputs ─────────────────────────────────────────────────────────────
st.sidebar.header("Market Inputs")
st.sidebar.caption("Defaults auto-populated from live CME prices (yfinance) and the latest CFTC/USDA reports")

st.sidebar.subheader("💰 Recent Cattle Prices")
le_1 = st.sidebar.number_input("Live Cattle — Last Month (¢/lb)",     80.0, 320.0, float(le_d[-1]), step=0.5)
le_2 = st.sidebar.number_input("Live Cattle — 2 Months Ago (¢/lb)",   80.0, 320.0, float(le_d[-2]), step=0.5)
le_3 = st.sidebar.number_input("Live Cattle — 3 Months Ago (¢/lb)",   80.0, 320.0, float(le_d[-3]), step=0.5)
gf_1 = st.sidebar.number_input("Feeder Cattle — Last Month (¢/lb)",   90.0, 420.0, float(gf_d[-1]), step=0.5)
gf_2 = st.sidebar.number_input("Feeder Cattle — 2 Months Ago (¢/lb)", 90.0, 420.0, float(gf_d[-2]), step=0.5)
gf_3 = st.sidebar.number_input("Feeder Cattle — 3 Months Ago (¢/lb)", 90.0, 420.0, float(gf_d[-3]), step=0.5)

st.sidebar.subheader("🌽 Feed Costs")
corn_roll6    = st.sidebar.number_input("Corn — 6-Month Avg (¢/bu)",         150.0, 900.0, float(prices["corn_roll6"]),    step=1.0)
corn_roll12   = st.sidebar.number_input("Corn — 12-Month Avg (¢/bu)",        150.0, 900.0, float(prices["corn_roll12"]),   step=1.0)
soymeal_roll3 = st.sidebar.number_input("Soybean Meal — 3-Month Avg ($/ton)", 200.0, 700.0, float(prices["soymeal_roll3"]), step=1.0)

st.sidebar.subheader("📊 Positioning & Packer Margin")
mm_net = st.sidebar.number_input(
    "CFTC Managed-Money Net Position (contracts)", -50000.0, 250000.0,
    float(local_data["mm_net"]), step=100.0,
    help="Speculative long minus short positions — from the CFTC Commitments of Traders report."
)
choice_cutout = st.sidebar.number_input(
    "Choice Beef Cutout Value ($/cwt)", 100.0, 400.0,
    float(local_data["choice_cutout"]), step=0.5,
    help="USDA AMS Choice 600-900 cutout value — a packer-margin signal."
)
cutout_spread_lag1 = st.sidebar.number_input(
    "Cutout Spread — Last Month ($/cwt)", -5.0, 60.0,
    float(local_data["cutout_spread_lag1"]), step=0.1,
    help="Choice minus Select cutout value, one month ago."
)

# ── Feature construction ───────────────────────────────────────────────────────
def build_feature_row():
    vals = {
        "live_cattle_price_lag1":   le_1,
        "feeder_cattle_price_lag1": gf_1,
        "feeder_cattle_price_lag2": gf_2,
        "soymeal_price_roll3":      soymeal_roll3,
        "mm_net":                   mm_net,
        "feeder_live_spread_roll3": np.mean([gf_1 - le_1, gf_2 - le_2, gf_3 - le_3]),
        "corn_price_roll12":        corn_roll12,
        "corn_price_roll6":         corn_roll6,
        "choice_cutout":            choice_cutout,
        "cutout_spread_lag1":       cutout_spread_lag1,
    }
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
    month_open_row = f"| Month Beginning Open Price | {month_open:.2f} ¢/lb |" if month_open else ""
    rows = "\n".join(filter(None, [
        f"| Live Cattle — Current Price | {le_1:.2f} ¢/lb |",
        month_open_row,
        f"| Live Cattle — 2 Months Ago | {le_2:.2f} ¢/lb |",
        f"| Feeder Cattle — Current Price | {gf_1:.2f} ¢/lb |",
        f"| Corn — 6-Mo Avg | {corn_roll6:.1f} ¢/bu |",
        f"| Corn — 12-Mo Avg | {corn_roll12:.1f} ¢/bu |",
    ]))
    st.markdown(f"| Input | Value |\n|-------|-------|\n{rows}")

with col2:
    st.markdown(f"""
| Input | Value |
|-------|-------|
| Soybean Meal — 3-Mo Avg | {soymeal_roll3:.1f} $/ton |
| CFTC Managed-Money Net | {mm_net:,.0f} contracts |
| Choice Cutout Value | {choice_cutout:.2f} $/cwt |
| Cutout Spread (Last Month) | {cutout_spread_lag1:.2f} $/cwt |
""")

st.markdown("---")
st.caption("Prices auto-fetched from CME via yfinance; positioning/cutout data from CFTC & USDA AMS · Model: Linear Regression | Data: USDA ERS + CME Futures + CFTC COT | CRISP-DM Capstone — Brian Place")
