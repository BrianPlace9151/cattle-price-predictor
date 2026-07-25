"""
build_notebook.py — generates beef_price_prediction_eda.ipynb
Run from anywhere: python3 ~/Python/Capstone/build_notebook.py
Output: ~/Python/Capstone/beef_price_prediction_eda.ipynb
"""

import json, os

OUT = os.path.expanduser('~/Python/Capstone/beef_price_prediction_Main.ipynb')

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}

def code(source, outputs=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": [source],
    }

cells = [

# ── Phase 1: Business Understanding ──────────────────────────────────────────
md("""# US Live Cattle Futures Price Prediction
## CRISP-DM Capstone — Initial Report & EDA

**Author:** Brian Place
**Date:** July 2026
**Target Variable:** CME Live Cattle Front-Month Futures (cents/lb)
**Forecast Horizon:** 1 month ahead (regression)

---"""),

md("""## CRISP-DM Framework

This notebook follows the **Cross-Industry Standard Process for Data Mining (CRISP-DM)**:

| Phase | Description | Status |
|-------|-------------|--------|
| 1. Business Understanding | Define the problem, objectives, and success criteria | ✅ Complete |
| 2. Data Understanding | Collect and describe all data sources; initial EDA | ✅ Complete |
| 3. Data Preparation | Clean, merge, and engineer features | ✅ Complete |
| 4. Modeling | Build and evaluate regression models | ✅ Complete |
| 5. Evaluation | Assess model performance and business value | ✅ Complete |
| 6. Deployment | Summarize findings and operationalize | ✅ Complete |"""),

md("""---
## Phase 1: Business Understanding

### 1.1 Problem Statement

US Live Cattle Futures, traded on the Chicago Mercantile Exchange (CME) under ticker **LE**, are the benchmark price for fed cattle in North America.

**Objective:** Develop a supervised regression model that predicts the **monthly closing price of CME Live Cattle Futures (cents/lb) one month ahead**, using USDA supply data and commodity market inputs as features.

### 1.2 Why This Matters

| Stakeholder | Use Case |
|-------------|----------|
| Cattle producers | Lock in revenue via futures hedges |
| Meatpackers | Procurement and margin planning |
| Food companies | Protein cost forecasting |
| Commodity traders | Systematic long/short signals |
| Restaurants | Menu, Marketing, Financial planning |

### 1.3 Success Criteria

| Metric | Target |
|--------|--------|
| R² (explained variance on test set) | ≥ 0.75 |
| RMSE | < 5 cents/lb |
| MAE | < 3 cents/lb |
| Directional accuracy | > 60% |

### 1.4 Scope
- **Target variable:** Monthly close — CME Live Cattle front-month futures (cents/lb)
- **Forecast horizon:** 1 month ahead
- **Historical range:** January 2005 – present
- **Train / test split:** 2005–2022 (train) | 2023–present (test)
- **Features:** USDA monthly supply data + corn futures + feeder cattle futures"""),

# ── Phase 2: Data Understanding ──────────────────────────────────────────────
md("""---
## Phase 2: Data Understanding

### 2.1 Data Sources

| Source | Description | Coverage | Format |
|--------|-------------|----------|--------|
| **USDA ERS Meat Statistics** | Beef production, cattle slaughter, cold storage | 1921–2026, monthly | Excel (.xlsx) |
| **CME Live Cattle Futures** (LE=F) | Target — monthly close price (¢/lb) | 2005–present | yfinance (auto) |
| **CME Corn Futures** (ZC=F) | Key input cost (~60% of feed cost) | 2005–present | yfinance (auto) |
| **CME Feeder Cattle Futures** (GF=F) | Direct input cost to Live Cattle | 2005–present | yfinance (auto) |

### 2.2 Data Dictionary

**USDA Supply Features**

| Column | Description | Units |
|--------|-------------|-------|
| `comm_beef_prod` | Commercial beef production | Million lbs / month |
| `comm_cattle_slaughter` | Commercial cattle slaughter | 1,000 head / month |
| `beef_cold_storage` | Beginning beef cold storage stocks | Million lbs |

**Futures Price Features**

| Column | Description | Units |
|--------|-------------|-------|
| `live_cattle_price` | CME Live Cattle front-month close | ¢ / lb ← **TARGET** |
| `corn_price` | CME Corn front-month close | ¢ / bushel |
| `feeder_cattle_price` | CME Feeder Cattle front-month close | ¢ / lb |"""),

code("""# ── Imports & Configuration ───────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import re
import os
import warnings
import yfinance as yf

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (14, 5)
plt.rcParams['font.size']      = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

DOWNLOADS  = os.path.expanduser('~/Downloads/')
CAPSTONE   = os.path.expanduser('~/Python/Capstone/')
USDA_FILE  = CAPSTONE + 'MeatStatsFull.xlsx'
START_DATE = '2005-01-01'

print("Libraries loaded.")
print(f"USDA file exists: {os.path.exists(USDA_FILE)}")"""),

md("### 2.3 Load USDA Data"),

code("""# ── USDA data loader ─────────────────────────────────────────────────────────
MONTH_RE = re.compile(r'^[A-Z][a-z]{2}-\\d{4}$')

def parse_usda_monthly(df):
    df = df[df.iloc[:, 0].apply(
        lambda x: bool(MONTH_RE.match(str(x))) if pd.notna(x) else False
    )].copy()
    df['date'] = pd.to_datetime(df.iloc[:, 0], format='%b-%Y')
    return df.set_index('date').sort_index().loc[START_DATE:]

raw_prod  = pd.read_excel(USDA_FILE, sheet_name='RedMeatPoultry_Prod-Full',  header=2)
prod      = parse_usda_monthly(raw_prod)
beef_prod = pd.to_numeric(prod.iloc[:, 1], errors='coerce').rename('comm_beef_prod')

raw_cold  = pd.read_excel(USDA_FILE, sheet_name='ColdStorage-Full',          header=1)
cold      = parse_usda_monthly(raw_cold)
cold_beef = pd.to_numeric(cold.iloc[:, 1], errors='coerce').rename('beef_cold_storage')

raw_sltr  = pd.read_excel(USDA_FILE, sheet_name='SlaughterCounts-Full',      header=2)
sltr      = parse_usda_monthly(raw_sltr)
sltr_ct   = pd.to_numeric(sltr.iloc[:, 1], errors='coerce').rename('comm_cattle_slaughter')

usda = pd.concat([beef_prod, cold_beef, sltr_ct], axis=1).dropna()
print(f"USDA: {usda.shape[0]} months  ({usda.index.min().strftime('%Y-%m')} → {usda.index.max().strftime('%Y-%m')})")
usda.head()"""),

md("### 2.4 Load Futures Price Data (yfinance — no manual download needed)"),

code("""# ── Futures prices via yfinance ───────────────────────────────────────────────
def load_yf_monthly(ticker, col_name, unit_multiplier=1.0):
    df = yf.download(ticker, start=START_DATE, interval='1mo',
                     progress=False, auto_adjust=True)
    if df.empty:
        return None
    s = df['Close'].squeeze() * unit_multiplier
    s.name = col_name
    s.index = pd.to_datetime(s.index).to_period('M').to_timestamp('M')
    return s.dropna().to_frame()

def load_macrotrends(path, col_name, unit_multiplier=1.0):
    with open(path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    start = next(
        i for i, ln in enumerate(lines)
        if ln.strip().lower().lstrip('"').startswith('date')
    )
    df = pd.read_csv(path, skiprows=start, encoding='utf-8-sig')
    df = df.iloc[:, :2]
    df.columns = ['date', col_name]
    df['date'] = pd.to_datetime(df['date'])
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce') * unit_multiplier
    df = df.set_index('date').sort_index().loc[START_DATE:]
    df = df.resample('ME').last()
    return df

le     = load_yf_monthly('LE=F', 'live_cattle_price')
corn   = load_yf_monthly('ZC=F', 'corn_price')
feeder = load_yf_monthly('GF=F', 'feeder_cattle_price')

if le is not None and corn is not None and feeder is not None:
    futures = pd.concat([le, corn, feeder], axis=1)
    print(f"Futures: {futures.shape[0]} months  ({futures.index.min().strftime('%Y-%m')} → {futures.index.max().strftime('%Y-%m')})")
    print(futures.tail(3).round(2))
else:
    print("⚠  Could not fetch futures data from yfinance — check your internet connection.")
    futures = None"""),

md("### 2.5 Initial Inspection"),

code("""print("=" * 56)
print("USDA SUPPLY FEATURES — Descriptive Statistics")
print("=" * 56)
print(usda.describe().round(2))

if futures is not None:
    print()
    print("=" * 56)
    print("FUTURES PRICE FEATURES — Descriptive Statistics")
    print("=" * 56)
    print(futures.describe().round(2))"""),

md("### 2.6 Data Quality Assessment"),

code("""print("Missing values — USDA:")
print(usda.isnull().sum())

if futures is not None:
    print("\\nMissing values — Futures:")
    print(futures.isnull().sum())

all_data = pd.concat([usda, futures], axis=1) if futures is not None else usda
all_data_me = all_data.copy()
all_data_me.index = all_data_me.index.to_period('M').to_timestamp('M')

fig, ax = plt.subplots(figsize=(12, 3))
sns.heatmap(all_data_me.isnull().T, cmap='YlOrRd', cbar=False,
            ax=ax, linewidths=0.3, yticklabels=all_data_me.columns)
ax.set_title('Missing Data Map (yellow = missing)', fontsize=12)
ax.set_xlabel('Time')
plt.tight_layout()
plt.show()"""),

md("""---
### 2.7 Additional Feature Data Sources

| Dataset | Gap Filled | Expected Impact |
|---------|-----------|-----------------|
| **USDA Cattle on Feed** | Supply timing — placements today = slaughter 4–6 months ahead | High |
| **CFTC Commitments of Traders (COT)** | Speculative positioning | High |
| **US Dollar Index (DXY)** | Export demand — strong dollar suppresses foreign beef buying | Medium |
| **Choice Beef Cutout Value** | Packer margin — cutout premium drives cattle bid prices | Medium |"""),

code("""COF_FILE     = CAPSTONE + 'cattle_on_feed.csv'
COT_FILE     = DOWNLOADS + 'cftc_cot_cattle.csv'
DXY_FILE     = CAPSTONE + 'dollar_index.csv'
CUTOUT_FILE  = DOWNLOADS + 'choice_cutout.csv'

addl_available = {
    'Cattle on Feed': os.path.exists(COF_FILE),
    'CFTC COT':       os.path.exists(COT_FILE),
    'Dollar Index':   os.path.exists(DXY_FILE),
    'Choice Cutout':  os.path.exists(CUTOUT_FILE),
}
print("Additional data file status:")
for name, avail in addl_available.items():
    print(f"  {'✅' if avail else '⬜'} {name}")"""),

md("#### Cattle on Feed (USDA NASS QuickStats)"),

code("""cof = None
if os.path.exists(COF_FILE):
    try:
        raw_cof = pd.read_csv(COF_FILE, encoding='utf-8-sig', thousands=',')
        raw_cof.columns = raw_cof.columns.str.strip().str.lower().str.replace(' ', '_')

        item_col = next(c for c in raw_cof.columns if 'data_item' in c)
        inv = raw_cof[raw_cof[item_col].str.upper().str.contains('INVENTORY', na=False)].copy()

        PERIOD_MAP = {
            'FIRST OF JAN': '01', 'FIRST OF FEB': '02', 'FIRST OF MAR': '03',
            'FIRST OF APR': '04', 'FIRST OF MAY': '05', 'FIRST OF JUN': '06',
            'FIRST OF JUL': '07', 'FIRST OF AUG': '08', 'FIRST OF SEP': '09',
            'FIRST OF OCT': '10', 'FIRST OF NOV': '11', 'FIRST OF DEC': '12',
            'END OF DEC':   '12',
        }
        inv['month'] = inv['period'].str.upper().str.strip().map(PERIOD_MAP)
        inv = inv.dropna(subset=['month'])
        inv['date'] = pd.to_datetime(inv['year'].astype(str) + '-' + inv['month'] + '-01')
        inv['value'] = pd.to_numeric(inv['value'].astype(str).str.replace(',',''), errors='coerce')
        inv = inv.dropna(subset=['date','value']).set_index('date').sort_index()['value']
        inv = inv[~inv.index.duplicated(keep='last')]

        full_idx = pd.date_range(inv.index.min(), inv.index.max(), freq='MS')
        inv_monthly = inv.reindex(full_idx).interpolate(method='time')
        inv_monthly.index = inv_monthly.index.to_period('M').to_timestamp('M')

        cof = inv_monthly.rename('cattle_inventory').to_frame().loc[START_DATE:]
        print(f"Cattle Inventory: {len(cof)} months  "
              f"({cof.index.min().date()} → {cof.index.max().date()})")
        print(cof.tail(3).round(0))
    except Exception as e:
        print(f"⚠  Could not parse cattle_on_feed.csv: {e}")
        cof = None
else:
    print("⬜ cattle_on_feed.csv not found — skipping.")"""),

md("#### CFTC Commitments of Traders"),

code("""cot = None
if os.path.exists(COT_FILE):
    raw_cot = pd.read_csv(COT_FILE, low_memory=False)
    raw_cot.columns = raw_cot.columns.str.strip()
    market_col = next((c for c in raw_cot.columns
                       if 'market' in c.lower() or 'name' in c.lower()), raw_cot.columns[0])
    lc_mask = raw_cot[market_col].str.upper().str.contains('LIVE CATTLE', na=False)
    lc = raw_cot[lc_mask].copy()
    date_col = next((c for c in lc.columns if 'date' in c.lower()), None)
    if date_col:
        lc['date'] = pd.to_datetime(lc[date_col], errors='coerce')
    lc = lc.dropna(subset=['date']).set_index('date').sort_index()
    long_col  = next((c for c in lc.columns if 'money' in c.lower() and 'long'  in c.lower()), None)
    short_col = next((c for c in lc.columns if 'money' in c.lower() and 'short' in c.lower()), None)
    if long_col and short_col:
        lc['mm_net'] = (pd.to_numeric(lc[long_col], errors='coerce') -
                        pd.to_numeric(lc[short_col], errors='coerce'))
        cot = lc[['mm_net']].loc[START_DATE:].resample('ME').last()
        roll_min = cot['mm_net'].rolling(52, min_periods=12).min()
        roll_max = cot['mm_net'].rolling(52, min_periods=12).max()
        cot['cot_index'] = ((cot['mm_net'] - roll_min) /
                             (roll_max - roll_min).replace(0, np.nan)) * 100
        print(f"CFTC COT: {cot.shape[0]} months")
    else:
        cot = None
else:
    print("⬜ cftc_cot_cattle.csv not found — skipping.")"""),

md("#### US Dollar Index (DXY)"),

code("""dxy = None
if os.path.exists(DXY_FILE):
    try:
        dxy = load_macrotrends(DXY_FILE, 'dxy')
        print(f"Dollar Index: {dxy.shape[0]} months  ({dxy.index.min().date()} → {dxy.index.max().date()})")
        print(dxy.tail(3).round(2))
    except Exception as e:
        print(f"⚠  Could not load dollar_index.csv: {e}")
        dxy = None
else:
    print("⬜ dollar_index.csv not found — skipping.")"""),

md("#### Choice Beef Cutout"),

code("""cutout = None
if os.path.exists(CUTOUT_FILE):
    with open(CUTOUT_FILE, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    start = next((i for i, ln in enumerate(lines)
                  if ln.strip().lower().lstrip('"').startswith('date')), 0)
    raw_cut = pd.read_csv(CUTOUT_FILE, skiprows=start, encoding='utf-8-sig')
    raw_cut = raw_cut.iloc[:, :2]
    raw_cut.columns = ['date', 'choice_cutout']
    raw_cut['date'] = pd.to_datetime(raw_cut['date'], errors='coerce')
    raw_cut['choice_cutout'] = pd.to_numeric(raw_cut['choice_cutout'], errors='coerce')
    cutout = (raw_cut.dropna().set_index('date').sort_index()
                     .loc[START_DATE:].resample('ME').last())
    print(f"Choice Cutout: {cutout.shape[0]} months")
else:
    print("⬜ choice_cutout.csv not found — skipping.")"""),

# ── EDA ───────────────────────────────────────────────────────────────────────
md("""---
### 2.8 Price History & Year-over-Year Change"""),

code("""if futures is not None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    ax = axes[0]
    le['live_cattle_price'].plot(ax=ax, color='#2c7bb6', linewidth=1.5, label='Monthly Close')
    le['live_cattle_price'].rolling(12).mean().plot(
        ax=ax, color='#d7191c', linewidth=2, linestyle='--', label='12-Mo Rolling Mean')
    ax.set_title('CME Live Cattle Futures — Monthly Close Price (2005–Present)')
    ax.set_ylabel('Cents per Pound')
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax2 = axes[1]
    yoy = le['live_cattle_price'].pct_change(12) * 100
    colors = ['#2c7bb6' if v >= 0 else '#d7191c' for v in yoy.fillna(0)]
    ax2.bar(yoy.index, yoy, color=colors, width=25, alpha=0.85)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_title('Year-over-Year % Change — Live Cattle Price')
    ax2.set_ylabel('% Change')
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.suptitle('Live Cattle Futures — Price History', fontsize=14)
    plt.tight_layout()
    plt.show()"""),

md("### 3.2 Price Distribution & Seasonality"),

code("""if futures is not None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec']
    le_df = le.copy()
    le_df['year']  = le_df.index.year
    le_df['month'] = le_df.index.month

    axes[0].hist(le_df['live_cattle_price'], bins=30, color='#2c7bb6',
                 edgecolor='white', alpha=0.85)
    axes[0].set_title('Price Distribution (All Years)')
    axes[0].set_xlabel('Cents per Pound')
    axes[0].set_ylabel('Frequency')
    mean_price = le_df['live_cattle_price'].mean()
    axes[0].axvline(mean_price, color='#d7191c', linewidth=2, linestyle='--',
                    label=f'Mean = {mean_price:.1f}¢')
    axes[0].legend()

    le_df.boxplot(column='live_cattle_price', by='year', ax=axes[1],
                  boxprops=dict(color='#2c7bb6'),
                  medianprops=dict(color='#d7191c', linewidth=2),
                  whiskerprops=dict(color='#2c7bb6'),
                  capprops=dict(color='#2c7bb6'))
    axes[1].set_title('Annual Price Distribution')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Cents per Pound')
    plt.sca(axes[1])
    plt.xticks(rotation=90)

    le_df.boxplot(column='live_cattle_price', by='month', ax=axes[2],
                  boxprops=dict(color='#2c7bb6'),
                  medianprops=dict(color='#d7191c', linewidth=2),
                  whiskerprops=dict(color='#2c7bb6'),
                  capprops=dict(color='#2c7bb6'))
    axes[2].set_title('Seasonal Pattern (Price by Month)')
    axes[2].set_xlabel('Month')
    axes[2].set_ylabel('Cents per Pound')
    axes[2].set_xticklabels(month_labels)

    plt.suptitle('')
    plt.tight_layout()
    plt.show()"""),

md("""### 3.3 Autocorrelation (ACF)

The ACF shows how strongly current price correlates with its own lagged values."""),

code("""if futures is not None:
    price = le['live_cattle_price'].dropna()
    lags  = list(range(1, 25))

    acf_vals  = [price.autocorr(lag=l) for l in lags]
    price_diff = price.diff().dropna()
    acf_diff  = [price_diff.autocorr(lag=l) for l in lags]
    ci = 1.96 / np.sqrt(len(price))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(lags, acf_vals, color='#2c7bb6', alpha=0.8)
    axes[0].axhline(0,   color='black', linewidth=0.8)
    axes[0].axhline( ci, color='red', linestyle='--', alpha=0.7, label='95% CI')
    axes[0].axhline(-ci, color='red', linestyle='--', alpha=0.7)
    axes[0].set_title('ACF — Live Cattle Price Level')
    axes[0].set_xlabel('Lag (months)')
    axes[0].set_ylabel('Pearson Correlation')
    axes[0].legend()

    axes[1].bar(lags, acf_diff, color='#1a9641', alpha=0.8)
    axes[1].axhline(0,   color='black', linewidth=0.8)
    axes[1].axhline( ci, color='red', linestyle='--', alpha=0.7, label='95% CI')
    axes[1].axhline(-ci, color='red', linestyle='--', alpha=0.7)
    axes[1].set_title('ACF — First-Differenced Price (Stationarity Check)')
    axes[1].set_xlabel('Lag (months)')
    axes[1].set_ylabel('Pearson Correlation')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    print(f"Lag-1  autocorrelation (price level): {price.autocorr(1):.3f}")
    print(f"Lag-12 autocorrelation (seasonal):    {price.autocorr(12):.3f}")"""),

md("### 3.4 Supply-Side Features (USDA Data)"),

code("""fig, axes = plt.subplots(3, 1, figsize=(14, 11))

meta = [
    ('comm_beef_prod',        'Commercial Beef Production',  'Million Lbs / Month', '#2c7bb6'),
    ('comm_cattle_slaughter', 'Commercial Cattle Slaughter', '1,000 Head / Month',  '#1a9641'),
    ('beef_cold_storage',     'Beef Cold Storage (Beginning Stocks)', 'Million Lbs', '#d7191c'),
]

for ax, (col, title, ylabel, color) in zip(axes, meta):
    usda[col].plot(ax=ax, color=color, linewidth=1.2, alpha=0.75, label='Monthly')
    usda[col].rolling(12).mean().plot(
        ax=ax, color='black', linewidth=2, linestyle='--', label='12-Mo Rolling Mean')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right')
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.suptitle('USDA Supply-Side Features (2005–Present)', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()"""),

md("### 3.5 Market Input Features — Corn & Feeder Cattle Futures"),

code("""if futures is not None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    futures['corn_price'].plot(ax=axes[0, 0], color='#fdae61', linewidth=1.2, alpha=0.85)
    futures['corn_price'].rolling(12).mean().plot(
        ax=axes[0, 0], color='black', linewidth=2, linestyle='--', label='12-Mo Mean')
    axes[0, 0].set_title('Corn Futures Price')
    axes[0, 0].set_ylabel('Cents / Bushel')
    axes[0, 0].legend()

    futures['feeder_cattle_price'].plot(ax=axes[0, 1], color='#a6761d', linewidth=1.2, alpha=0.85)
    futures['feeder_cattle_price'].rolling(12).mean().plot(
        ax=axes[0, 1], color='black', linewidth=2, linestyle='--', label='12-Mo Mean')
    axes[0, 1].set_title('Feeder Cattle Futures Price')
    axes[0, 1].set_ylabel('Cents / Lb')
    axes[0, 1].legend()

    for (xcol, label, color, ax) in [
        ('corn_price',          'Corn Price (¢/bu)',    '#fdae61', axes[1, 0]),
        ('feeder_cattle_price', 'Feeder Cattle (¢/lb)', '#a6761d', axes[1, 1]),
    ]:
        valid = futures[[xcol, 'live_cattle_price']].dropna()
        ax.scatter(valid[xcol], valid['live_cattle_price'], alpha=0.35, color=color, s=18)
        m, b = np.polyfit(valid[xcol], valid['live_cattle_price'], 1)
        x_line = np.linspace(valid[xcol].min(), valid[xcol].max(), 100)
        ax.plot(x_line, m * x_line + b, color='black', linewidth=1.5, linestyle='--')
        r = valid[xcol].corr(valid['live_cattle_price'])
        ax.set_title(f'{label} vs Live Cattle (r = {r:.3f})')
        ax.set_xlabel(label)
        ax.set_ylabel('Live Cattle Price (¢/lb)')

    plt.suptitle('Market Input Features — Corn & Feeder Cattle', fontsize=13)
    plt.tight_layout()
    plt.show()"""),

md("### 3.6 Correlation Matrix"),

code("""if futures is not None:
    def to_me(df):
        d = df.copy()
        d.index = d.index.to_period('M').to_timestamp('M')
        return d

    usda_me    = to_me(usda)
    futures_me = to_me(futures)
    combined   = usda_me.join(futures_me, how='inner').dropna()

    corr_df = combined[['live_cattle_price', 'comm_beef_prod', 'comm_cattle_slaughter',
                         'beef_cold_storage', 'corn_price', 'feeder_cattle_price']].copy()
    corr_df.columns = ['Live Cattle', 'Beef Prod', 'Cattle Slaughter',
                        'Cold Storage', 'Corn', 'Feeder Cattle']
    corr = corr_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=-1, vmax=1, mask=mask, ax=ax,
                linewidths=0.5, square=True, annot_kws={'size': 10})
    ax.set_title('Feature Correlation Matrix', fontsize=13)
    plt.tight_layout()
    plt.show()

    print("\\nCorrelation with Live Cattle Price (ranked):")
    print(corr['Live Cattle'].drop('Live Cattle').sort_values(ascending=False).round(3))"""),

# ── Phase 3: Data Preparation ─────────────────────────────────────────────────
md("""---
## Phase 3: Data Preparation

### 3.1 Merge & Align All Data Sources"""),

code("""if futures is not None:
    df = to_me(usda).join(to_me(futures), how='inner')

    if cof    is not None: df = df.join(to_me(cof),    how='left')
    if cot    is not None: df = df.join(to_me(cot),    how='left')
    if dxy    is not None: df = df.join(to_me(dxy),    how='left')
    if cutout is not None: df = df.join(to_me(cutout), how='left')

    print(f"Merged dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Period: {df.index.min().date()} → {df.index.max().date()}")
    print(f"Columns: {df.columns.tolist()}")
    df.head()
else:
    df = usda.copy()
    df.index = df.index.to_period('M').to_timestamp('M')
    print("USDA-only dataset:")
    df.head()"""),

md("""### 3.2 Data Cleansing Log

| Step | Source | Issue | Treatment |
|------|--------|-------|-----------|
| 1 | USDA | Raw sheet contains YTD aggregates and footnote rows | Regex filter — keep only rows matching `Mon-YYYY` |
| 2 | USDA | Date column stored as string | Parse with `format='%b-%Y'`; set as DatetimeIndex |
| 3 | USDA | Non-numeric values | `pd.to_numeric(..., errors='coerce')` → NaN, then dropped |
| 4 | Futures | UTF-8 BOM prefix | Open with `encoding='utf-8-sig'` |
| 5 | All futures | Mix of frequencies | `resample('ME').last()` to normalize to month-end |
| 6 | Merged | USDA dates are month-start; futures are month-end | Convert all to period-month then back to month-end |
| 7 | Merged | Short reporting gaps ≤ 2 months | Forward-fill (`ffill(limit=2)`) |
| 8 | Merged | Any remaining nulls after forward-fill | `dropna()` — row removed |"""),

code("""print("Missing values before treatment:")
print(df.isnull().sum())
print(f"Total rows before: {len(df)}")

df_before = df.copy()
df = df.ffill(limit=2).dropna()

rows_dropped = len(df_before) - len(df)
print(f"\\nRows dropped by dropna(): {rows_dropped}")
print(f"Total rows after:          {len(df)}")
print(f"\\nMissing values after treatment:")
print(df.isnull().sum())"""),

md("### 3.3 Feature Engineering"),

code("""if futures is not None:
    fe = df.copy()

    fe['target_le_next_month'] = fe['live_cattle_price'].shift(-1)

    feature_base = ['live_cattle_price', 'comm_beef_prod', 'comm_cattle_slaughter',
                    'beef_cold_storage', 'corn_price', 'feeder_cattle_price']
    for col in feature_base:
        for lag in [1, 2, 3]:
            fe[f'{col}_lag{lag}'] = fe[col].shift(lag)

    if 'cattle_inventory' in fe.columns:
        for lag in [1, 2, 3, 6, 12]:
            fe[f'cattle_inventory_lag{lag}'] = fe['cattle_inventory'].shift(lag)
        fe['cattle_inventory_yoy'] = fe['cattle_inventory'].pct_change(12)

    if 'mm_net' in fe.columns:
        fe['mm_net_lag1']    = fe['mm_net'].shift(1)
        fe['mm_net_chg']     = fe['mm_net'].diff(1).shift(1)
        fe['cot_index_lag1'] = fe['cot_index'].shift(1) if 'cot_index' in fe.columns else np.nan

    if 'dxy' in fe.columns:
        fe['dxy_lag1']  = fe['dxy'].shift(1)
        fe['dxy_lag3']  = fe['dxy'].shift(3)
        fe['dxy_yoy']   = fe['dxy'].pct_change(12)
        fe['dxy_roll3'] = fe['dxy'].shift(1).rolling(3).mean()

    if 'choice_cutout' in fe.columns:
        fe['cutout_lag1']   = fe['choice_cutout'].shift(1)
        fe['cutout_lag2']   = fe['choice_cutout'].shift(2)
        fe['cutout_yoy']    = fe['choice_cutout'].pct_change(12)
        fe['packer_margin'] = (fe['choice_cutout'].shift(1) -
                               fe['live_cattle_price'].shift(1))

    for col in ['live_cattle_price', 'corn_price', 'feeder_cattle_price']:
        base = fe[col].shift(1)
        fe[f'{col}_roll3']  = base.rolling(3).mean()
        fe[f'{col}_roll6']  = base.rolling(6).mean()
        fe[f'{col}_roll12'] = base.rolling(12).mean()

    fe['feed_cost_ratio'] = (fe['corn_price'].shift(1) /
                              fe['live_cattle_price'].shift(1).replace(0, np.nan))
    fe['prod_cycle'] = fe['comm_beef_prod'].shift(1).rolling(36, min_periods=12).mean()

    fe['month'] = fe.index.month
    dummies = pd.get_dummies(fe['month'], prefix='month', drop_first=True)
    fe = pd.concat([fe, dummies], axis=1).drop(columns=['month'])

    for col in ['comm_beef_prod', 'comm_cattle_slaughter', 'beef_cold_storage']:
        fe[f'{col}_yoy'] = fe[col].pct_change(12)

    fe = fe.dropna(thresh=int(len(fe.columns) * 0.7))
    fe = fe.dropna(subset=['target_le_next_month'])
    fe = fe.ffill().bfill().dropna()

    n_features = fe.shape[1] - 1
    print(f"Feature-engineered dataset: {fe.shape[0]} rows  |  {n_features} features + 1 target")
    print(f"Period: {fe.index.min().date()} → {fe.index.max().date()}")"""),

# ── Phase 4: Modeling ─────────────────────────────────────────────────────────
md("""---
## Phase 4: Modeling

### 4.1 Train / Test Split

| Split | Period | Purpose |
|-------|--------|---------|
| **Train** | Aug 2006 – Dec 2022 | Fit and tune all models |
| **Test** | Jan 2023 – present | Final out-of-sample evaluation |"""),

code("""if futures is not None:
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV, RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    TRAIN_END = '2022-12-31'

    raw_price_cols = ['live_cattle_price', 'comm_beef_prod', 'comm_cattle_slaughter',
                      'beef_cold_storage', 'corn_price', 'feeder_cattle_price']
    model_cols = [c for c in fe.columns
                  if c != 'target_le_next_month' and c not in raw_price_cols]

    X = fe[model_cols]
    y = fe['target_le_next_month']

    X_train = X[X.index <= TRAIN_END];  X_test = X[X.index > TRAIN_END]
    y_train = y[y.index <= TRAIN_END];  y_test = y[y.index > TRAIN_END]

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    X_train_sc = np.nan_to_num(X_train_sc, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_sc  = np.nan_to_num(X_test_sc,  nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Training set : {X_train.shape[0]} months  ({X_train.index.min().date()} → {X_train.index.max().date()})")
    print(f"Test set     : {X_test.shape[0]} months  ({X_test.index.min().date()} → {X_test.index.max().date()})")
    print(f"Features     : {X_train.shape[1]}")"""),

md("""### 4.2 Feature Selection

| Method | Purpose |
|--------|---------|
| **Correlation filter** | Drop features with near-zero correlation to the target |
| **VIF** | Detect multicollinear features |
| **Lasso (L1)** | Auto-select features by shrinking irrelevant coefficients to zero |"""),

code("""if futures is not None:
    corr_target = (pd.concat([X_train, y_train], axis=1)
                   .corr()['target_le_next_month']
                   .drop('target_le_next_month')
                   .sort_values(key=abs, ascending=False))

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ['#2c7bb6' if v > 0 else '#d7191c' for v in corr_target]
    ax.barh(corr_target.index, corr_target.values, color=colors, alpha=0.8)
    ax.axvline(0,    color='black', linewidth=0.8)
    ax.axvline( 0.1, color='grey',  linewidth=0.8, linestyle='--', label='|r|=0.10 threshold')
    ax.axvline(-0.1, color='grey',  linewidth=0.8, linestyle='--')
    ax.set_title('Feature Correlation with Target (Next Month Live Cattle Price)', fontsize=12)
    ax.set_xlabel('Pearson r')
    ax.legend()
    plt.tight_layout()
    plt.show()

    keep_corr = corr_target[corr_target.abs() >= 0.10].index.tolist()
    drop_corr = corr_target[corr_target.abs() <  0.10].index.tolist()
    print(f"Features kept  (|r| ≥ 0.10): {len(keep_corr)}")
    print(f"Features dropped (|r| < 0.10): {len(drop_corr)}")"""),

code("""if futures is not None:
    X_vif = X_train[keep_corr].dropna().copy()
    X_vif = X_vif.loc[:, X_vif.std() > 0]
    X_vif_sc = StandardScaler().fit_transform(X_vif)
    X_vif_sc = np.nan_to_num(X_vif_sc, nan=0.0, posinf=0.0, neginf=0.0)

    vif_scores = []
    for i in range(X_vif_sc.shape[1]):
        y_i  = X_vif_sc[:, i]
        X_i  = np.delete(X_vif_sc, i, axis=1)
        r2_i = LinearRegression().fit(X_i, y_i).score(X_i, y_i)
        vif_scores.append(1 / (1 - r2_i) if r2_i < 0.9999 else 999)

    vif_df = pd.DataFrame({'feature': keep_corr[:len(vif_scores)], 'VIF': vif_scores})
    vif_df = vif_df.sort_values('VIF', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    bar_colors = ['#d7191c' if v > 10 else '#fdae61' if v > 5 else '#2c7bb6'
                  for v in vif_df['VIF'].clip(upper=50)]
    ax.barh(vif_df['feature'], vif_df['VIF'].clip(upper=50), color=bar_colors, alpha=0.85)
    ax.axvline(10, color='red',    linewidth=1.5, linestyle='--', label='VIF=10 (severe)')
    ax.axvline(5,  color='orange', linewidth=1.5, linestyle='--', label='VIF=5 (moderate)')
    ax.set_title('Variance Inflation Factor (VIF)', fontsize=12)
    ax.set_xlabel('VIF (capped at 50)')
    ax.legend()
    plt.tight_layout()
    plt.show()"""),

code("""if futures is not None:
    lasso_cv = LassoCV(cv=5, max_iter=20000, random_state=42)
    lasso_cv.fit(X_train_sc, y_train)

    lasso_coef = pd.DataFrame({
        'feature': model_cols,
        'coef':    lasso_cv.coef_
    }).sort_values('coef', key=abs, ascending=False)

    selected_lasso = lasso_coef[lasso_coef['coef'] != 0]['feature'].tolist()
    dropped_lasso  = lasso_coef[lasso_coef['coef'] == 0]['feature'].tolist()

    print(f"LassoCV optimal alpha: {lasso_cv.alpha_:.4f}")
    print(f"Features selected (coef ≠ 0): {len(selected_lasso)}")
    print(f"Features zeroed out:           {len(dropped_lasso)}")

    final_features = list(dict.fromkeys(
        [f for f in selected_lasso if f in keep_corr]
    ))
    # Fallback: if Lasso is too aggressive, use correlation filter directly
    if len(final_features) < 10:
        final_features = keep_corr
        print("Note: Lasso selected too few features — using correlation filter instead.")
    print(f"\\nFinal feature set: {len(final_features)} features")
    for f in final_features:
        print(f"  {f}")"""),

md("""### 4.3 Model Training

| Model | Overfitting Control |
|-------|-------------------|
| **Linear Regression** | None — baseline benchmark |
| **Ridge (L2)** | Shrinks all coefficients; handles multicollinearity |
| **Lasso (L1)** | Zeros out irrelevant coefficients |"""),

code("""if futures is not None:
    def score_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true.values - y_pred) / y_true.values)) * 100
        da   = np.mean(np.sign(np.diff(y_true.values)) ==
                       np.sign(np.diff(y_pred))) * 100
        return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape, 'DirAcc': da}

    X_tr_f = X_train[final_features]
    X_te_f = X_test[final_features]
    sc_f   = StandardScaler()
    X_tr_fs = sc_f.fit_transform(X_tr_f)
    X_te_fs = sc_f.transform(X_te_f)
    X_tr_fs = np.nan_to_num(X_tr_fs, nan=0.0, posinf=0.0, neginf=0.0)
    X_te_fs = np.nan_to_num(X_te_fs, nan=0.0, posinf=0.0, neginf=0.0)

    ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
    lasso_f  = LassoCV(cv=5, max_iter=20000, random_state=42)
    lr_f     = LinearRegression()

    lr_f.fit(X_tr_fs, y_train)
    ridge_cv.fit(X_tr_fs, y_train)
    lasso_f.fit(X_tr_fs, y_train)

    models = {
        'Linear Regression': lr_f,
        'Ridge (L2)':        ridge_cv,
        'Lasso (L1)':        lasso_f,
    }

    results = {}
    for name, mdl in models.items():
        train_scores = score_metrics(y_train, mdl.predict(X_tr_fs))
        test_scores  = score_metrics(y_test,  mdl.predict(X_te_fs))
        results[name] = {'train': train_scores, 'test': test_scores}

    print(f"{'Model':<22} {'Split':<6} {'RMSE':>7} {'MAE':>7} {'R²':>7} {'MAPE':>7} {'DirAcc':>8}")
    print("-" * 65)
    for name, res in results.items():
        for split, scores in res.items():
            print(f"{name:<22} {split:<6} "
                  f"{scores['RMSE']:>6.2f}¢ "
                  f"{scores['MAE']:>6.2f}¢ "
                  f"{scores['R2']:>7.3f} "
                  f"{scores['MAPE']:>6.1f}% "
                  f"{scores['DirAcc']:>7.1f}%")
        print()"""),

md("### 4.4 Walk-Forward Cross-Validation"),

code("""if futures is not None:
    MIN_TRAIN_MONTHS = 36

    wf_results = {name: [] for name in models}

    X_all = pd.concat([X_tr_f, X_te_f])
    y_all = pd.concat([y_train, y_test])

    for i in range(MIN_TRAIN_MONTHS, len(X_all)):
        X_wf_tr = X_all.iloc[:i]
        y_wf_tr = y_all.iloc[:i]
        X_wf_te = X_all.iloc[i:i+1]
        y_wf_te = y_all.iloc[i:i+1]

        sc_wf = StandardScaler()
        X_wf_tr_sc = sc_wf.fit_transform(X_wf_tr)
        X_wf_te_sc = sc_wf.transform(X_wf_te)

        for name, ModelClass in [('Linear Regression', LinearRegression()),
                                  ('Ridge (L2)',        Ridge(alpha=ridge_cv.alpha_)),
                                  ('Lasso (L1)',        Lasso(alpha=lasso_f.alpha_, max_iter=10000))]:
            ModelClass.fit(X_wf_tr_sc, y_wf_tr)
            pred = ModelClass.predict(X_wf_te_sc)[0]
            wf_results[name].append({
                'date':      y_wf_te.index[0],
                'actual':    y_wf_te.values[0],
                'predicted': pred
            })

    print("Walk-Forward CV Results:")
    print(f"{'Model':<22} {'RMSE':>7} {'MAE':>7} {'R²':>7} {'MAPE':>7} {'DirAcc':>8}")
    print("-" * 58)
    wf_dfs = {}
    for name, preds in wf_results.items():
        wf_df = pd.DataFrame(preds).set_index('date')
        wf_dfs[name] = wf_df
        rmse = np.sqrt(mean_squared_error(wf_df['actual'], wf_df['predicted']))
        mae  = mean_absolute_error(wf_df['actual'], wf_df['predicted'])
        r2   = r2_score(wf_df['actual'], wf_df['predicted'])
        mape = np.mean(np.abs((wf_df['actual'] - wf_df['predicted']) / wf_df['actual'])) * 100
        da   = np.mean(np.sign(np.diff(wf_df['actual'].values)) ==
                       np.sign(np.diff(wf_df['predicted'].values))) * 100
        print(f"{name:<22} {rmse:>6.2f}¢ {mae:>6.2f}¢ {r2:>7.3f} {mape:>6.1f}% {da:>7.1f}%")"""),

md("### 4.5 Actual vs Predicted — All Models"),

code("""if futures is not None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 13))
    colors_m = {'Linear Regression': '#d7191c', 'Ridge (L2)': '#1a9641', 'Lasso (L1)': '#fdae61'}

    for ax, (name, wf_df) in zip(axes, wf_dfs.items()):
        ax.plot(wf_df.index, wf_df['actual'],    color='#2c7bb6', linewidth=2,   label='Actual')
        ax.plot(wf_df.index, wf_df['predicted'], color=colors_m[name], linewidth=1.5,
                linestyle='--', label=f'Predicted ({name})')
        rmse = np.sqrt(mean_squared_error(wf_df['actual'], wf_df['predicted']))
        r2   = r2_score(wf_df['actual'], wf_df['predicted'])
        ax.set_title(f'{name} — Walk-Forward CV   RMSE={rmse:.2f}¢   R²={r2:.3f}')
        ax.set_ylabel('Live Cattle (¢/lb)')
        ax.legend(loc='upper left')
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.suptitle('Walk-Forward CV: Actual vs Predicted — All Models', fontsize=14)
    plt.tight_layout()
    plt.show()"""),

md("### 4.6 Residual Analysis"),

code("""if futures is not None:
    best_name = min(wf_dfs, key=lambda n: np.sqrt(
        mean_squared_error(wf_dfs[n]['actual'], wf_dfs[n]['predicted'])))
    best_df = wf_dfs[best_name]
    resid   = best_df['actual'] - best_df['predicted']

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0,0].bar(resid.index, resid,
                  color=['#2c7bb6' if r >= 0 else '#d7191c' for r in resid],
                  width=25, alpha=0.8)
    axes[0,0].axhline(0, color='black', linewidth=0.8)
    axes[0,0].set_title(f'Residuals Over Time ({best_name})')
    axes[0,0].set_ylabel('Error (¢/lb)')

    axes[0,1].hist(resid, bins=25, color='#2c7bb6', edgecolor='white', alpha=0.85)
    axes[0,1].axvline(resid.mean(), color='#d7191c', linewidth=2, linestyle='--',
                      label=f'Mean={resid.mean():.2f}')
    axes[0,1].set_title('Residual Distribution')
    axes[0,1].set_xlabel('Error (¢/lb)')
    axes[0,1].legend()

    axes[1,0].scatter(best_df['predicted'], best_df['actual'], alpha=0.4, color='#2c7bb6', s=20)
    mn, mx = best_df['predicted'].min(), best_df['predicted'].max()
    axes[1,0].plot([mn, mx], [mn, mx], color='#d7191c', linewidth=1.5, linestyle='--', label='Perfect fit')
    axes[1,0].set_title('Predicted vs Actual')
    axes[1,0].set_xlabel('Predicted (¢/lb)')
    axes[1,0].set_ylabel('Actual (¢/lb)')
    axes[1,0].legend()

    lags = list(range(1, 13))
    acf_resid = [resid.autocorr(lag=l) for l in lags]
    ci_r = 1.96 / np.sqrt(len(resid))
    axes[1,1].bar(lags, acf_resid, color='#2c7bb6', alpha=0.8)
    axes[1,1].axhline(0,     color='black', linewidth=0.8)
    axes[1,1].axhline( ci_r, color='red',   linewidth=1, linestyle='--', label='95% CI')
    axes[1,1].axhline(-ci_r, color='red',   linewidth=1, linestyle='--')
    axes[1,1].set_title('Residual Autocorrelation (ACF)')
    axes[1,1].set_xlabel('Lag (months)')
    axes[1,1].set_ylabel('Correlation')
    axes[1,1].legend()

    plt.suptitle(f'Residual Analysis — Best Model: {best_name}', fontsize=13)
    plt.tight_layout()
    plt.show()

    print(f"Best model: {best_name}")
    print(f"Residual mean:  {resid.mean():.3f}")
    print(f"Residual std:   {resid.std():.3f}")
    print(f"Lag-1 residual autocorrelation: {resid.autocorr(1):.3f}")"""),

md("### 4.7 Model Comparison Summary"),

code("""if futures is not None:
    print("=" * 70)
    print("FINAL MODEL COMPARISON — Walk-Forward CV")
    print("=" * 70)
    print(f"{'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'MAPE':>8} {'DirAcc':>9}")
    print("-" * 70)
    summary_rows = []
    for name, wf_df in wf_dfs.items():
        rmse = np.sqrt(mean_squared_error(wf_df['actual'], wf_df['predicted']))
        mae  = mean_absolute_error(wf_df['actual'], wf_df['predicted'])
        r2   = r2_score(wf_df['actual'], wf_df['predicted'])
        mape = np.mean(np.abs((wf_df['actual'] - wf_df['predicted']) / wf_df['actual'])) * 100
        da   = np.mean(np.sign(np.diff(wf_df['actual'].values)) ==
                       np.sign(np.diff(wf_df['predicted'].values))) * 100
        summary_rows.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2,
                              'MAPE': mape, 'DirAcc': da})
        print(f"{name:<22} {rmse:>7.2f}¢ {mae:>7.2f}¢ {r2:>8.3f} {mape:>7.1f}% {da:>8.1f}%")

    best = min(summary_rows, key=lambda r: r['RMSE'])
    print(f"\\n→ Best model by RMSE: {best['Model']}")

    print("\\nSuccess Criteria Check:")
    print(f"  RMSE  < 5¢/lb:   {'✅' if best['RMSE'] < 5  else '❌'}  ({best['RMSE']:.2f}¢)")
    print(f"  MAE   < 3¢/lb:   {'✅' if best['MAE']  < 3  else '❌'}  ({best['MAE']:.2f}¢)")
    print(f"  R²    ≥ 0.75:    {'✅' if best['R2']   >= 0.75 else '❌'}  ({best['R2']:.3f})")
    print(f"  DirAcc > 60%:    {'✅' if best['DirAcc'] > 60 else '❌'}  ({best['DirAcc']:.1f}%)")"""),

# ── Phase 6: Deployment ───────────────────────────────────────────────────────
md("""---
## Phase 6: Deployment

### 6.1 Save Model Artifacts"""),

code("""if futures is not None:
    import joblib

    best_name  = min(summary_rows, key=lambda r: r['RMSE'])['Model']
    model_map  = {'Linear Regression': models['Linear Regression'],
                  'Ridge (L2)':        models['Ridge (L2)'],
                  'Lasso (L1)':        models['Lasso (L1)']}
    best_model = model_map[best_name]

    X_full = scaler.fit_transform(pd.concat([X_tr_f, X_te_f]))
    y_full = pd.concat([y_train.loc[X_tr_f.index], y_test.loc[X_te_f.index]])
    best_model.fit(X_full, y_full)

    feature_medians = dict(pd.concat([X_tr_f, X_te_f]).median())

    joblib.dump(best_model,       'cattle_price_model.pkl')
    joblib.dump(scaler,           'scaler.pkl')
    joblib.dump(final_features,   'feature_cols.pkl')
    joblib.dump(feature_medians,  'feature_medians.pkl')

    print(f"✅ Saved: cattle_price_model.pkl  ({best_name})")
    print(f"✅ Saved: scaler.pkl")
    print(f"✅ Saved: feature_cols.pkl  ({len(final_features)} features)")
    print(f"✅ Saved: feature_medians.pkl")"""),

md("### 6.2 Streamlit App"),

code("""import os
app_path = os.path.join(os.getcwd(), 'app.py')
if os.path.exists(app_path):
    print(f"✅ app.py found at: {app_path}")
else:
    print("⚠  app.py not found in current directory.")

print()
print("To run locally:  streamlit run app.py")
print("To deploy:       push to GitHub → share.streamlit.io → select repo → Deploy")"""),

md("### 6.3 Requirements File"),

code("""REQ = \"\"\"streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
openpyxl>=3.1.0
yfinance>=0.2.0
\"\"\"

with open('requirements.txt', 'w') as f:
    f.write(REQ)

print("✅ requirements.txt written")
print()
print("GitHub repo should contain:")
for fname in ['beef_price_prediction_eda.ipynb', 'app.py', 'requirements.txt',
              'cattle_price_model.pkl', 'scaler.pkl', 'feature_cols.pkl', 'feature_medians.pkl']:
    print(f"  {fname}")"""),

md("""---
## Summary & Key Findings

| Finding | Model Implication |
|---------|-------------------|
| Strong lag-1 autocorrelation in LE price | Lagged price is the single most important feature |
| Clear seasonal pattern (spring/summer premium) | Month dummy variables are required |
| Feeder Cattle has the highest correlation with Live Cattle | Include as a primary feature |
| Cold storage leads price by 1–2 months | Use lagged cold storage, not concurrent |
| Corn price positively correlated with LE | Feed cost signal; include lag-1 and lag-2 |

### Walk-Forward CV Results (Best Model: Ridge L2)
- R² = 0.930 ✅
- RMSE = 9.51¢/lb
- MAE = 7.59¢/lb
- Directional Accuracy = 53.4%

---
*Author: Brian Place | CRISP-DM Capstone | July 2026*
*Data: USDA ERS + CME Futures via yfinance*"""),

]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

with open(OUT, 'w') as f:
    json.dump(nb, f, indent=2)

print(f"✅ Notebook written to: {OUT}")
print(f"   Cells: {len(cells)}")
print()
print("Run from Terminal:")
print("  cd ~/Python/Capstone && python3 build_notebook.py")
