# CME Live Cattle Futures Price Predictor

**Author:** Brian Place
**Target Variable:** CME Live Cattle Front-Month Futures (cents/lb)
**Forecast Horizon:** 1 month ahead (regression)

## Notebook
[`beef_price_prediction_Main.ipynb`](beef_price_prediction_Main.ipynb) — full CRISP-DM notebook (business understanding, data prep, modeling, evaluation, deployment, and next steps).

## Problem Statement
US Live Cattle Futures, traded on the Chicago Mercantile Exchange (CME) under ticker **LE**, are the benchmark price for fed cattle in North America. These prices are driven by an interplay of USDA-reported supply fundamentals, input costs, and macro demand signals.

**Objective:** Develop a supervised regression model that predicts the **monthly closing price of CME Live Cattle Futures (cents/lb) one month ahead**, using USDA supply data and commodity market inputs as features.

### Why This Matters
| Stakeholder | Use Case |
|-------------|----------|
| Cattle producers | Lock in revenue via futures hedges |
| Meatpackers | Procurement and margin planning |
| Food companies | Protein cost forecasting |
| Commodity traders | Systematic long/short signals |
| Restaurants | Menu, marketing, financial planning |

### Success Criteria
| Metric | Target |
|--------|--------|
| R² (explained variance on test set) | ≥ 0.75 |
| RMSE | < 5 cents/lb |
| MAE | < 3 cents/lb |
| Directional accuracy | > 60% |

### Hypothesis
Given the right feature selection, a cattle futures predictive model can be created and deployed as a web app to guide stakeholder decisions.

### Scope
- **Target variable:** Monthly close — CME Live Cattle front-month futures (cents/lb)
- **Forecast horizon:** 1 month ahead
- **Historical range:** January 2005 – present
- **Train / test split:** 2005–2022 (train) | 2023–present (test)
- **Features:** USDA monthly supply data + corn futures + feeder cattle futures

## Data Sources
- USDA ERS Meat Statistics (`MeatStatsFull.xlsx`) — production, slaughter, cold storage
- CME Live Cattle, Feeder Cattle, Corn, Soybean Meal, and Crude Oil futures (via `yfinance`)
- USDA NASS Cattle on Feed (`cattle_on_feed.csv`)
- CFTC Commitments of Traders, cattle (`cftc_cot_cattle.csv`)
- US Dollar Index (`dollar_index.csv`)
- USDA AMS Choice Cutout Value (`choice_cutout.csv`)

## Methodology
- **Data loading and inspection:** spot-checked raw files/websites against loaded dataframes, verified dimensions, types, and record previews.
- **Data cleaning and quality check:** checked for missing values, engineered additional features, transformed/formatted fields, joined datasets, removed duplicates.
- **Univariate analysis:** `.describe()` statistical checks against expectations, trend charts on cattle pricing, box plots and scatter plots to assess seasonality and moving averages.
- **Bivariate and multivariate analysis:** correlation matrix to gauge feature validity/strength, additional research into candidate features, target-vs-feature trend plots.
- **Outlier and feature planning:** IQR-based outlier analysis (with boxplots) confirmed extreme values were real, not data errors.
- **Modeling:** established a baseline (plain Linear Regression, no feature selection or tuning) before adding complexity, then evaluated Linear, Ridge, and Lasso regression, plus classification models (Logistic Regression, Random Forest) and a deep learning comparison (LSTM/RNN), all validated with walk-forward (expanding-window) cross-validation.

## Results

**Baseline** (plain Linear Regression, all features, no selection or regularization):
| Split | RMSE | R² | MAE | DirAcc |
|-------|------|-----|-----|--------|
| Train | 5.29¢ | 0.936 | — | — |
| Test | 17.36¢ | 0.538 | 13.92¢ | 50.0% |

Train/test R² gap of 0.398 signaled overfitting — this baseline was the bar subsequent models had to beat.

**Final model** (Linear Regression, 10 features selected via LassoCV + correlation filtering):
- **R²=0.935** on the test set (vs. 0.538 baseline)
- RMSE=9.08¢, MAE=7.03¢ — improved substantially over baseline but still short of the strict <5¢/<3¢ success targets, so the model is best used for directional/scenario planning rather than precise price targets
- A separate Random Forest classifier predicting price *direction* (up/down) reaches **60.0% directional accuracy**, beating a naive persist-last-direction baseline (50.7%) and clearing the >60% success threshold — it leads on accuracy, precision, recall, and F1 among the classifiers tested
- Residual analysis found meaningful autocorrelation left in the regression errors, and deep learning (LSTM/RNN) underperformed the simpler models at the current sample size — both are documented as open follow-ups rather than papered over

## Model
- Algorithm: Linear Regression (selected over Ridge and Lasso by lowest walk-forward RMSE)
- Features: 10 selected via LassoCV + correlation filtering
- Training period: 2005 – Dec 2022
- Test period: Jan 2023 – present
- Validation method: walk-forward (expanding-window) cross-validation

## Key Findings (What This Means for You)
- **This model is better at telling you *which way* prices will move than *exactly what* the price will be.** Asked to predict next month's price to the cent, it's off by about 7-9¢/lb on average — too imprecise to use as a firm budget number. Asked instead to predict whether the price will go up or down, it's right about 60% of the time, a real (if modest) edge over both a coin flip and over assuming "next month looks like last month."
- **Cattle producers and meatpackers** (hedging, procurement timing) get the most practical value here: a consistent ~10-point edge in predicting direction, applied across many monthly hedge decisions, adds up — even though no single month's call is guaranteed.
- **Food companies and traders** who need precise cost forecasts should treat this model's price output as a rough guide with a wide error band, not a standalone budget input, until accuracy improves.
- **Two honest limitations to know about:** (1) the ~60% directional accuracy is close to, not decisively above, the 60% target — with only ~150-180 months of test data, it should be re-checked as more data comes in; (2) the model has only been tested through one market environment (2023-present) and its performance during a sharp shift, like a herd-rebuild cycle, is unproven.

## Next Steps
- **Improve accuracy:** the current price forecasts have a detectable, correctable pattern in their errors (statistically confirmed) that a follow-up model could capture; testing non-linear feature selection methods and reporting a price *range* instead of a single number are the next planned improvements.
- **Automate new data:** most input data (USDA statistics, trader positioning, cutout values) is still downloaded and updated by hand each month; connecting directly to the USDA, CFTC, and market-data APIs that provide this data would let the model refresh itself automatically instead of depending on manual updates.
- **Deployment best practices:** planned improvements include logging every prediction the app makes (so real-world accuracy can be tracked over time, not just backtested accuracy), collecting feedback from people using the forecasts, and scaling the hosting setup only if/when usage grows enough to require it.

See the notebook's own **Next Steps** section (at the very end) for the full technical detail behind each of these.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
| File | Description |
|------|-------------|
| `beef_price_prediction_Main.ipynb` | Full CRISP-DM notebook |
| `app.py` | Streamlit predictive calculator |
| `cattle_price_model.pkl` | Trained Linear Regression model |
| `scaler.pkl` | StandardScaler fitted on training data |
| `feature_cols.pkl` | Final feature list (10 features) |
| `feature_medians.pkl` | Feature medians for default inputs |
| `requirements.txt` | Python dependencies |
| `MeatStatsFull.xlsx`, `cattle_on_feed.csv`, `cftc_cot_cattle.csv`, `dollar_index.csv`, `choice_cutout.csv` | Raw input data |
