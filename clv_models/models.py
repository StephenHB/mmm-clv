"""
models.py

Modular BG/NBD and Gamma-Gamma model classes for CLV analysis using the lifetimes package.
Includes demonstration code and documentation.
"""

import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter

class BGNBDModel:
    """
    Modular BG/NBD model for customer transaction frequency prediction.
    Uses the lifetimes.BetaGeoFitter under the hood.
    """
    def __init__(self, penalizer_coef=0.0):
        self.model = BetaGeoFitter(penalizer_coef=penalizer_coef)
        self.fitted = False

    def fit(self, frequency, recency, T):
        """Fit the BG/NBD model to customer data."""
        self.model.fit(frequency, recency, T)
        self.fitted = True
        return self

    def predict(self, t, frequency, recency, T):
        """Predict expected number of transactions in time t for each customer."""
        if not self.fitted:
            raise RuntimeError("Model must be fit before prediction.")
        return self.model.conditional_expected_number_of_purchases_up_to_time(t, frequency, recency, T)

    @property
    def params_(self):
        return self.model.params_

class GammaGammaCLVModel:
    """
    Modular Gamma-Gamma model for customer monetary value prediction.
    Uses the lifetimes.GammaGammaFitter under the hood.
    """
    def __init__(self, penalizer_coef=0.0):
        self.model = GammaGammaFitter(penalizer_coef=penalizer_coef)
        self.fitted = False

    def fit(self, frequency, monetary_value):
        """Fit the Gamma-Gamma model to customer data."""
        self.model.fit(frequency, monetary_value)
        self.fitted = True
        return self

    def predict(self, frequency, monetary_value):
        """Predict the expected average transaction value for each customer."""
        if not self.fitted:
            raise RuntimeError("Model must be fit before prediction.")
        return self.model.conditional_expected_average_profit(frequency, monetary_value)

    @property
    def params_(self):
        return self.model.params_

# --- Demonstration Example ---
def demo_clv_modeling():
    """
    Demonstrate fitting and predicting with BG/NBD and Gamma-Gamma models using synthetic data.
    """
    import numpy as np
    # Generate synthetic data
    n_customers = 1000
    data = pd.DataFrame({
        'customer_id': np.arange(1, n_customers + 1),
        'frequency': np.random.poisson(2, n_customers),
        'recency': np.random.uniform(0, 30, n_customers),
        'T': np.random.uniform(30, 60, n_customers),
        'monetary_value': np.random.uniform(10, 200, n_customers),
    })
    # Fit BG/NBD
    bg_nbd = BGNBDModel().fit(data['frequency'], data['recency'], data['T'])
    print("BG/NBD params:", bg_nbd.params_)
    # Predict expected purchases in next 30 days
    data['predicted_purchases_30d'] = bg_nbd.predict(30, data['frequency'], data['recency'], data['T'])
    # Fit Gamma-Gamma
    gamma_gamma = GammaGammaCLVModel().fit(data['frequency'], data['monetary_value'])
    print("Gamma-Gamma params:", gamma_gamma.params_)
    # Predict expected average profit
    data['predicted_avg_profit'] = gamma_gamma.predict(data['frequency'], data['monetary_value'])
    # Calculate CLV (30 days)
    data['predicted_clv_30d'] = data['predicted_purchases_30d'] * data['predicted_avg_profit']
    print("\nSample CLV predictions:\n", data[['customer_id', 'predicted_clv_30d']].head())
    return data

if __name__ == "__main__":
    demo_clv_modeling() 