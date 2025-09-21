"""
data_preprocess_utils.py

Shared utilities for data preprocessing across different models.
"""

import pandas as pd
import numpy as np


def load_retail_data(file_path, encoding='ISO-8859-1'):
    """Load the Online Retail II dataset from CSV."""
    return pd.read_csv(file_path, encoding=encoding)


def clean_transaction_data(df):
    """
    Clean transaction data by removing cancellations, missing IDs, and invalid values.
    
    Args:
        df: Raw transaction DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Remove cancelled orders (Invoice starts with 'C')
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


def create_customer_summary(df, customer_id_col='Customer ID', 
                          datetime_col='InvoiceDate', 
                          monetary_value_col='TotalPrice'):
    """
    Create customer-level summary statistics.
    
    Args:
        df: Cleaned transaction DataFrame
        customer_id_col: Column name for customer ID
        datetime_col: Column name for transaction datetime
        monetary_value_col: Column name for monetary value
        
    Returns:
        Customer summary DataFrame
    """
    from lifetimes.utils import summary_data_from_transaction_data
    
    # Convert datetime if needed
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    # Create summary using lifetimes utility
    summary = summary_data_from_transaction_data(
        df,
        customer_id_col=customer_id_col,
        datetime_col=datetime_col,
        monetary_value_col=monetary_value_col,
        observation_period_end=df[datetime_col].max()
    )
    
    # Remove customers with zero or negative monetary_value
    summary = summary[summary['monetary_value'] > 0]
    
    return summary


def validate_data_quality(df, required_columns=None):
    """
    Validate data quality and report issues.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Dictionary with validation results
    """
    if required_columns is None:
        required_columns = ['Invoice', 'StockCode', 'Description', 'Quantity', 
                          'InvoiceDate', 'Price', 'Customer ID', 'Country']
    
    validation_results = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'required_columns_present': all(col in df.columns for col in required_columns)
    }
    
    return validation_results
