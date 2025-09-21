"""
visualization.py

Basic visual analysis of the raw Online Retail II data and the final CLV output.
Generates histograms and scatter plots for key features and CLV predictions.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths (update if needed)
RAW_DATA_PATH = "/Users/stephenzhang/Downloads/clv_data/Year_2010_2011.csv"
CLV_RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'clv_results_sample.csv')

PLOT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
HTML_REPORT_PATH = os.path.join(os.path.dirname(__file__), 'visualization_report.html')
os.makedirs(PLOT_DIR, exist_ok=True)

PLOT_PATHS = []
PROB_ALIVE_HEATMAP_PATH = os.path.join(PLOT_DIR, 'probability_alive_heatmap.png')

def plot_histogram(data, column, title, xlabel, bins=50, fname=None):
    plt.figure(figsize=(8, 5))
    sns.histplot(data[column], bins=bins, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.tight_layout()
    if fname:
        path = os.path.join(PLOT_DIR, fname)
        plt.savefig(path)
        PLOT_PATHS.append((title, path))
    plt.close()


def plot_scatter(data, x, y, title, xlabel, ylabel, fname=None):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=data[x], y=data[y], alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if fname:
        path = os.path.join(PLOT_DIR, fname)
        plt.savefig(path)
        PLOT_PATHS.append((title, path))
    plt.close()


def visualize_raw_data():
    """Visualize basic statistics of the raw transaction data."""
    df = pd.read_csv(RAW_DATA_PATH, encoding='ISO-8859-1')
    print("Raw data loaded. Shape:", df.shape)
    plot_histogram(df, 'Quantity', 'Distribution of Quantity per Transaction', 'Quantity', fname='quantity_hist.png')
    plot_histogram(df, 'Price', 'Distribution of Unit Price', 'Unit Price (£)', fname='price_hist.png')
    # Removed Customer ID histogram as it's not meaningful


def visualize_clv_results():
    """Visualize the distribution and relationships in the CLV results."""
    df = pd.read_csv(CLV_RESULTS_PATH)
    print("CLV results loaded. Shape:", df.shape)
    plot_histogram(df, 'monetary_value', 'Distribution of Average Monetary Value', 'Monetary Value (£)', fname='clv_monetary_hist.png')
    plot_histogram(df, 'frequency', 'Distribution of Frequency', 'Frequency', fname='clv_frequency_hist.png')
    plot_histogram(df, 'recency', 'Distribution of Recency', 'Recency (days)', fname='clv_recency_hist.png')
    plot_histogram(df, 'predicted_clv_30d', 'Distribution of Predicted 30-Day CLV', 'Predicted CLV (£)', fname='clv_predicted_hist.png')
    plot_scatter(df, 'frequency', 'monetary_value', 'Frequency vs. Monetary Value', 'Frequency', 'Monetary Value (£)', fname='clv_freq_vs_monetary.png')


def compute_rfm(df):
    """Compute RFM (Recency, Frequency, Monetary) metrics from raw transaction data."""
    # Convert InvoiceDate to datetime if not already
    df = df.copy()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    # Reference date for recency (last transaction date in dataset)
    reference_date = df['InvoiceDate'].max()
    # Group by Customer ID
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
    return rfm

def visualize_rfm():
    """Visualize the distribution of RFM metrics."""
    df = pd.read_csv(RAW_DATA_PATH, encoding='ISO-8859-1')
    # Preprocess for RFM: remove cancellations, missing IDs, negative/zero values
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    df = df[df['Customer ID'].notnull()]
    df = df[df['Quantity'] > 0]
    df = df[df['Price'] > 0]
    df['TotalPrice'] = df['Quantity'] * df['Price']
    rfm = compute_rfm(df)
    print("RFM table shape:", rfm.shape)
    plot_histogram(rfm, 'Recency', 'Distribution of Recency (RFM)', 'Recency (days)', fname='rfm_recency_hist.png')
    plot_histogram(rfm, 'Frequency', 'Distribution of Frequency (RFM)', 'Frequency', fname='rfm_frequency_hist.png')
    plot_histogram(rfm, 'Monetary', 'Distribution of Monetary Value (RFM)', 'Monetary Value (£)', fname='rfm_monetary_hist.png')


def generate_html_report():
    """Generate an HTML report embedding all saved plots and including descriptions."""
    with open(HTML_REPORT_PATH, 'w') as f:
        f.write('<html><head><title>CLV Analysis Visualization Report</title></head><body>\n')
        f.write('<h1>CLV Analysis Visualization Report</h1>\n')
        # Raw Data Section
        f.write('<h2>Raw Data Analysis</h2>\n')
        f.write('<p>The following plots show the distribution of transaction quantities and unit prices in the raw dataset. These help identify outliers, data quality issues, and the overall sales distribution.</p>\n')
        for title, path in PLOT_PATHS:
            if 'Quantity' in title or 'Unit Price' in title:
                f.write(f'<h3>{title}</h3>\n')
                f.write(f'<img src="plots/{os.path.basename(path)}" style="max-width:700px;"><br><br>\n')
        # RFM Section
        f.write('<h2>RFM (Recency, Frequency, Monetary) Analysis</h2>\n')
        f.write('<p>RFM analysis segments customers based on how recently (Recency), how often (Frequency), and how much (Monetary) they purchase. These distributions help identify high-value and at-risk customers.</p>\n')
        for title, path in PLOT_PATHS:
            if '(RFM)' in title:
                f.write(f'<h3>{title}</h3>\n')
                f.write(f'<img src="plots/{os.path.basename(path)}" style="max-width:700px;"><br><br>\n')
        # CLV Section
        f.write('<h2>CLV Model Results</h2>\n')
        f.write('<p>The following plots show the distribution of model-estimated monetary value, frequency, recency, and predicted 30-day CLV. The scatter plot visualizes the relationship between frequency and monetary value, highlighting different customer segments.</p>\n')
        for title, path in PLOT_PATHS:
            if 'CLV' in title or 'Frequency vs. Monetary Value' in title:
                f.write(f'<h3>{title}</h3>\n')
                f.write(f'<img src="plots/{os.path.basename(path)}" style="max-width:700px;"><br><br>\n')
        # Probability Alive Heatmap Section
        f.write('<h2>Probability Customer is Alive (BG/NBD Heatmap)</h2>\n')
        f.write('<p>This heatmap shows the probability that a customer is still "alive" (active) as a function of their purchase frequency and recency, based on the BG/NBD model. Higher frequency and lower recency (recent purchases) correspond to higher probabilities of being active. This visualization helps target retention efforts.</p>\n')
        f.write(f'<img src="plots/{os.path.basename(PROB_ALIVE_HEATMAP_PATH)}" style="max-width:700px;"><br><br>\n')
        f.write('</body></html>\n')
    print(f"HTML report generated: {HTML_REPORT_PATH}")


def main():
    print("Visualizing raw data...")
    visualize_raw_data()
    print("Visualizing RFM metrics...")
    visualize_rfm()
    print("Visualizing CLV results...")
    visualize_clv_results()
    # Ensure the probability_alive heatmap is generated and saved before report
    from clv_models.models import BGNBDModel, plot_probability_alive_heatmap
    import pandas as pd
    from lifetimes.utils import summary_data_from_transaction_data
    df = pd.read_csv(RAW_DATA_PATH, encoding='ISO-8859-1')
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    df = df[df['Customer ID'].notnull()]
    df = df[df['Quantity'] > 0]
    df = df[df['Price'] > 0]
    df['TotalPrice'] = df['Quantity'] * df['Price']
    summary = summary_data_from_transaction_data(df, customer_id_col='Customer ID', datetime_col='InvoiceDate', monetary_value_col='TotalPrice', observation_period_end=df['InvoiceDate'].max())
    summary = summary[summary['monetary_value'] > 0]
    bg_nbd = BGNBDModel().fit(summary['frequency'], summary['recency'], summary['T'])
    plot_probability_alive_heatmap(bg_nbd, max_frequency=10, max_recency=30, T=30, save_path=PROB_ALIVE_HEATMAP_PATH)
    generate_html_report()

if __name__ == "__main__":
    main() 