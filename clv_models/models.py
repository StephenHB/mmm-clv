"""
models.py

Modular BG/NBD and Gamma-Gamma model classes for CLV analysis using the lifetimes package.
Includes demonstration code and documentation.
"""

import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

    def probability_alive(self, frequency, recency, T):
        """
        Calculate the probability that a customer is still "alive" (active) given their frequency, recency, and T.
        Returns a numpy array of probabilities.
        """
        if not self.fitted:
            raise RuntimeError("Model must be fit before prediction.")
        return self.model.conditional_probability_alive(frequency, recency, T)

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

def plot_probability_alive_heatmap(bgnbd_model, max_frequency=10, max_recency=30, T=30, save_path=None):
    """
    Plot a heatmap of probability_alive for different frequency and recency values, similar to Zhihu article.
    """
    freq_range = np.arange(0, max_frequency + 1)
    rec_range = np.arange(0, max_recency + 1)
    heatmap = np.zeros((len(freq_range), len(rec_range)))
    for i, f in enumerate(freq_range):
        for j, r in enumerate(rec_range):
            heatmap[i, j] = bgnbd_model.probability_alive(f, r, T)
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap,
        xticklabels=[str(x) for x in rec_range],
        yticklabels=[str(y) for y in freq_range],
        cmap='YlGnBu'
    )
    plt.xlabel('Recency (days)')
    plt.ylabel('Frequency')
    plt.title('Probability Customer is Alive (BG/NBD)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    demo_clv_modeling() 