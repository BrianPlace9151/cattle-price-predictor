# CME Live Cattle Futures Price Predictor

A 1-month ahead Live Cattle (LE) futures price forecasting model built as a CRISP-DM data science capstone.

## Features
- Predicts next month's CME Live Cattle Futures price (¢/lb)
- Uses USDA supply data, futures prices, corn feed costs, and macroeconomic inputs
- Interactive Streamlit web app for real-time forecasting

## Data Sources
- USDA ERS Meat Statistics (beef production, slaughter, cold storage)
- CME Live Cattle & Feeder Cattle Futures (Investing.com)
- Corn prices (Macrotrends)
- USDA NASS Cattle Inventory
- US Dollar Index (Investing.com)

## Model
- Algorithm: Lasso Regression (L1 regularization)
- Features: 41 selected via LassoCV + correlation filter
- Training period: Aug 2006 – Dec 2022
- Test period: Jan 2023 – Apr 2026

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
| File | Description |
|------|-------------|
| `beef_price_prediction_eda.ipynb` | Full CRISP-DM notebook |
| `app.py` | Streamlit predictive calculator |
| `cattle_price_model.pkl` | Trained Lasso model |
| `scaler.pkl` | StandardScaler fitted on training data |
| `feature_cols.pkl` | Final feature list (41 features) |
| `feature_medians.pkl` | Feature medians for default inputs |
| `requirements.txt` | Python dependencies |

*Author: Brian Place | July 2026*
