# clv_models

## Data Description: Online Retail II Dataset

This dataset contains all the transactions occurring for a UK-based and registered, non-store online retail between 01/12/2009 and 09/12/2011. The company mainly sells unique all-occasion gift-ware. Many customers of the company are wholesalers.

### Variables
- **InvoiceNo**: Invoice number. Nominal. A 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter 'c', it indicates a cancellation.
- **StockCode**: Product (item) code. Nominal. A 5-digit integral number uniquely assigned to each distinct product.
- **Description**: Product (item) name. Nominal.
- **Quantity**: The quantities of each product (item) per transaction. Numeric.
- **InvoiceDate**: Invoice date and time. Numeric. The day and time when a transaction was generated.
- **UnitPrice**: Unit price. Numeric. Product price per unit in sterling (£).
- **CustomerID**: Customer number. Nominal. A 5-digit integral number uniquely assigned to each customer.
- **Country**: Country name. Nominal. The name of the country where a customer resides.

## Project Setup

1. **Create and activate a virtual environment:**
   ```sh
   python3 -m venv marketing_env
   source marketing_env/bin/activate
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Download the dataset:**
   Place the CSV file (e.g., `Year_2010_2011.csv`) in a known location (update the path in scripts if needed).

## Running the CLV Analysis POC

To run the end-to-end CLV analysis and generate results:
```sh
python clv_models/clv_analysis_poc.py
```
- This will load and preprocess the data, fit BG/NBD and Gamma-Gamma models, and output sample CLV predictions to `clv_results_sample.csv`.

## Generating Visualizations and HTML Report

To generate all visualizations and a comprehensive HTML report:
```sh
python clv_models/visualization.py
```
- This will create plots for raw data, RFM metrics, CLV results, and the probability_alive heatmap.
- The report will be saved as `clv_models/visualization_report.html` (open in your browser).
- All plot images are saved in `clv_models/plots/` (ignored by git).

### Probability Alive Heatmap
The probability_alive heatmap visualizes the likelihood that a customer is still "alive" (active) as a function of their purchase frequency and recency, based on the BG/NBD model. This helps identify which customers are most likely to return and can guide retention strategies.

## Linting the Codebase

To check code quality and style using ruff:
```sh
ruff check .
```

## Models Used for CLV

### BG/NBD Model
The Beta-Geometric/Negative Binomial Distribution (BG/NBD) model is used to predict the number of future transactions for a customer based on their purchase history.

### Gamma-Gamma Model
The Gamma-Gamma model is used to estimate the average monetary value of a customer's transactions, assuming the monetary value does not vary with transaction frequency.

These models are implemented using the [lifetimes](https://github.com/CamDavidsonPilon/lifetimes) Python package.

## References
- [UCI ML Repository: Online Retail II Data Set](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)
- [BG/NBD and Gamma-Gamma models for CLV](https://benalexkeen.com/bg-nbd-model-for-customer-base-analysis-in-python/)
- [lifetimes Python package](https://github.com/CamDavidsonPilon/lifetimes) 