"""
clv_analysis_poc.py

Proof-of-concept for Customer Lifetime Value (CLV) analysis using the Online Retail II dataset.
Follows the project PRD: loads data, preprocesses for CLV, fits BG/NBD and Gamma-Gamma models, and outputs sample predictions.
"""

import os
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter

# Path to the CSV dataset (update if needed)
DATA_PATH = "/Users/stephenzhang/Downloads/clv_data/Year_2010_2011.csv"


def load_data(path=DATA_PATH):
    """Load the Online Retail II dataset from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please ensure the CSV file is present.")
    df = pd.read_csv(path, encoding='ISO-8859-1')
    return df


def preprocess_data(df):
    """Preprocess the data for CLV analysis (remove cancellations, missing IDs, negative quantities, etc.)."""
    df = df.copy()
    # Remove cancelled orders
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    # Remove missing Customer ID
    df = df[df['Customer ID'].notnull()]
    # Remove negative or zero quantities
    df = df[df['Quantity'] > 0]
    # Remove negative or zero prices
    df = df[df['Price'] > 0]
    # Calculate total price
    df['TotalPrice'] = df['Quantity'] * df['Price']
    return df


def create_summary_table(df, observation_period_end=None):
    """Create summary table for lifetimes modeling (frequency, recency, T, monetary_value)."""
    from lifetimes.utils import summary_data_from_transaction_data
    if observation_period_end is None:
        observation_period_end = df['InvoiceDate'].max()
    summary = summary_data_from_transaction_data(
        df,
        customer_id_col='Customer ID',
        datetime_col='InvoiceDate',
        monetary_value_col='TotalPrice',
        observation_period_end=observation_period_end
    )
    # Remove customers with zero or negative monetary_value
    summary = summary[summary['monetary_value'] > 0]
    return summary


def fit_models(summary):
    """Fit BG/NBD and Gamma-Gamma models to the summary data."""
    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(summary['frequency'], summary['recency'], summary['T'])
    ggf = GammaGammaFitter(penalizer_coef=0.0)
    ggf.fit(summary['frequency'], summary['monetary_value'])
    return bgf, ggf


def predict_clv(bgf, ggf, summary, time=30):
    """Predict CLV for each customer over a given time period (default: 30 days)."""
    clv = ggf.customer_lifetime_value(
        bgf,
        summary['frequency'],
        summary['recency'],
        summary['T'],
        summary['monetary_value'],
        time=time,  # prediction period in days
        freq='D',   # frequency of T (days)
        discount_rate=0.01
    )
    return clv


def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df):,} rows.")
    print("Preprocessing data...")
    df_clean = preprocess_data(df)
    print(f"After cleaning: {len(df_clean):,} rows.")
    print("Creating summary table...")
    summary = create_summary_table(df_clean)
    print(f"Summary table: {summary.shape[0]:,} customers.")
    print("Fitting models...")
    bgf, ggf = fit_models(summary)
    print("BG/NBD params:", bgf.params_)
    print("Gamma-Gamma params:", ggf.params_)
    print("Predicting CLV for next 30 days...")
    clv = predict_clv(bgf, ggf, summary, time=30)
    print("\nSample CLV predictions:\n", clv.head())
    # Optionally, save results
    summary['predicted_clv_30d'] = clv
    summary.to_csv(os.path.join(os.path.dirname(__file__), 'clv_results_sample.csv'))
    print("Results saved to clv_results_sample.csv")

if __name__ == "__main__":
    main() 