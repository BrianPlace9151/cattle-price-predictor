US Live Cattle Futures Price Predictor

Author: Brian Place Date: July 2026 Target Variable: CME Live Cattle Front-Month Futures (cents/lb) Forecast Horizon: 1 month ahead (regression)

Problem Statement

US Live Cattle Futures, traded on the Chicago Mercantile Exchange (CME) under ticker LE, are the benchmark price for fed cattle in North America. These prices are driven by an interplay of USDA-reported supply fundamentals, input costs, and macro demand signals.

Objective

Develop a supervised regression model that predicts the monthly closing price of CME Live Cattle Futures (cents/lb) one month ahead, using USDA supply data and commodity market inputs as features.

Why This Matters

Stakeholder Use Case
Cattle producers Lock in revenue via futures hedges
Meatpackers Procurement and margin planning
Food companies Protein cost forecasting
Commodity traders Systematic long/short signals
Restaurants Menu, Marketing, Financial planning

Success Criteria

Metric Target R² (explained variance on test set) ≥ 0.75
RMSE < 5 cents/lb
MAE < 3 cents/lb
Directional accuracy > 60%

Hypothesis:

I believe given the right feature selection; a cattle futures predictive model can be created and deployed as a web app to guide stakeholder decisions.

Scope

Target variable: Monthly close — CME Live Cattle front-month futures (cents/lb)
Forecast horizon: 1 month ahead
Historical range: January 2005 – present
Train / test split: 2005–2022 (train) | 2023–present (test)
Features: USDA monthly supply data + corn futures + feeder cattle futures

Features

Predicts next month's CME Live Cattle Futures price (¢/lb)
Uses USDA supply data, futures prices, corn feed costs, and macroeconomic inputs
Interactive Streamlit web app for real-time forecasting
Data Sources

USDA ERS Meat Statistics (beef production, slaughter, cold storage) (ers.usda.gove) (Excel(.xlsx)
CME Live Cattle & Feeder Cattle Futures (yfinance (auto))
Corn prices (yfinance (auto))
USDA NASS Cattle Inventory (quickstats.nass.usda.gov)
US Dollar Index (fred.stlouis.fed.org)
Data Loading and Inspection

Loaded data from source and performed a spot check analysis between rawfile/website and header rows on DF
Check dimensions: look at total rows and columns
Inspect types: verify numerical, categorical, and date fields
Preview records: view the first and last few rows.
Data Cleaning and Quality Check

Looked for missing values
Performed feature engineering analysis to determine data additions needed
Performed data transformation and formatting where needed
Combined data sets with joins
Removed duplicates
Univariate Analysis (Single Variable)

Performed .describe statistical analysis to validate data and compare vs. expectations
Created trend charts on cattle pricing
Created box plots and scatter plots to determine seasonality and moving averages
Bivariate and Multivariate Analysis (Multiple Variables)

Created correlation matrix to gauge validity and strength of selected features
Did additional research to see if any additional features should be added
Plotted target trended pricing vs. feature trend pricing
Outlier and Feature Planning

Used IQR to perform outlier analysis with aid of boxplots and determined data was real
Model

Before adding complexity, we establish a baseline: a plain linear regression fit on all features with no selection or tuning was run
Various regression algorithms (Linear, Lasso, and Ridge) will be deployed to achieve the best accuracy scores (RMSE, MAE R²)
Initial Findings

Based on trend chart comparisons and correlation matrix two of my features (Feeder Cattle and Corn) are strong and positive predictors for live cattle price

Baseline: Linear Regression (all features, no regulariztion)
Train RMSE : 5.29¢ Train R² : 0.936
Test RMSE : 17.36¢ Test R² : 0.538
Test MAE : 13.92¢
Test DirAcc: 50.0%

⚠ Train/test R² gap = 0.398 — signs of overfitting. Feature selection and regularization (next section) should improve this.

This score is the bar to beat. Any more complex model must outperform it.
