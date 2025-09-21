# MMM-CLV: Customer Lifetime Value Analysis

A comprehensive Customer Lifetime Value (CLV) analysis framework using Lifetimes (BG/NBD + Gamma-Gamma) models.

## Project Structure

```
mmm-clv/
├── src/                          # Production-ready code
│   ├── main.ipynb               # Main execution notebook
│   ├── data_preprocess/         # Data preprocessing modules
│   │   ├── data_preprocess_utils.py  # Shared utilities
│   │   ├── lifetimes/           # Lifetimes-specific preprocessing
│   └── model/                   # Core model algorithms
│       ├── model_utils.py       # Shared model utilities
│       ├── lifetimes/           # Lifetimes model implementation
│       │   ├── models.py        # BG/NBD and Gamma-Gamma models
│       │   ├── clv_analysis_poc.py  # Analysis pipeline
│       │   └── visualization.py # Visualization tools
├── data/                        # Data storage
│   ├── lifetimes/              # Lifetimes model data
├── notebooks/                   # Development and testing
├── test/                        # Unit tests
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Features

### Lifetimes Model
- **BG/NBD Model**: Predicts customer transaction frequency
- **Gamma-Gamma Model**: Estimates customer monetary value
- **Probability Alive**: Calculates customer activity probability
- **Visualization**: Comprehensive plots and HTML reports


## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv marketing_env
source marketing_env/bin/activate  # On Windows: marketing_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify setup (optional)
python -c "import sys; sys.path.insert(0, '.'); from src.model.lifetimes.models import BGNBDModel; print('✅ Setup complete!')"
```

### 2. Prepare Data

Place your transaction data CSV file in `data/lifetimes/` directory.

Expected CSV columns:
- `Invoice`: Invoice number (cancellations start with 'C')
- `StockCode`: Product code
- `Description`: Product description
- `Quantity`: Quantity purchased
- `InvoiceDate`: Transaction date
- `Price`: Unit price
- `Customer ID`: Customer identifier
- `Country`: Customer country

### 3. Run Analysis

#### Option A: Complete Analysis Script (Recommended)
```bash
# Run complete CLV analysis with all outputs
python src/run_clv_analysis.py
```

#### Option B: Jupyter Notebook (Interactive)
```bash
jupyter notebook src/main.ipynb
```

#### Option C: Individual Components
```bash
# Lifetimes model analysis only
python src/model/lifetimes/clv_analysis_poc.py

# Generate visualizations only
python src/model/lifetimes/visualization.py
```

## Model-Specific Usage

### Lifetimes Model

```python
from src.model.lifetimes.models import BGNBDModel, GammaGammaCLVModel
from src.data_preprocess.data_preprocess_utils import load_retail_data, clean_transaction_data

# Load and clean data
df = load_retail_data('path/to/data.csv')
df_clean = clean_transaction_data(df)

# Create customer summary
summary = create_customer_summary(df_clean)

# Fit models
bg_nbd = BGNBDModel().fit(summary['frequency'], summary['recency'], summary['T'])
gamma_gamma = GammaGammaCLVModel().fit(summary['frequency'], summary['monetary_value'])

# Predict CLV
clv = gamma_gamma.customer_lifetime_value(bg_nbd, ...)
```


## Output Files

After running the analysis, you'll find the following outputs:

### Data Files
- `data/lifetimes/clv_results_complete.csv`: Complete CLV analysis results with predictions
- `data/lifetimes/clv_results_sample.csv`: Sample results from previous runs

### Visualizations
- `data/lifetimes/plots/`: Directory containing all generated plots
  - `probability_alive_heatmap.png`: Customer activity probability heatmap
  - `clv_*_hist.png`: Distribution histograms for CLV metrics
  - `rfm_*_hist.png`: RFM analysis plots
  - `*_scatter.png`: Scatter plots for relationship analysis

### Reports
- `src/model/lifetimes/visualization_report.html`: Comprehensive HTML report with all visualizations and analysis

## Visualization and Reporting

The framework generates comprehensive visualizations including:

- **Raw Data Analysis**: Distribution plots for quantities, prices
- **RFM Analysis**: Recency, Frequency, Monetary value distributions
- **CLV Results**: Model predictions and customer segments
- **Probability Alive Heatmap**: Customer activity probability visualization
- **HTML Reports**: Comprehensive analysis reports

## Development

### Code Quality
```bash
# Lint codebase
ruff check .

# Run tests (when implemented)
pytest test/
```

### Adding New Models

1. Create model directory: `src/model/your_model/`
2. Implement model class following the interface
3. Add preprocessing utilities: `src/data_preprocess/your_model/`
4. Update `src/main.ipynb` to include your model
5. Add data directory: `data/your_model/`

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computing
- lifetimes: BG/NBD and Gamma-Gamma models
- matplotlib/seaborn: Visualization
- scikit-learn: Machine learning utilities
- jupyter: Notebook environment

## References

- [Lifetimes Documentation](https://github.com/CamDavidsonPilon/lifetimes)
- [BG/NBD Model Paper](https://www.brucehardie.com/papers/018/fader_et_al_mksc_05.pdf)
- [Gamma-Gamma Model Paper](https://www.brucehardie.com/notes/025/gamma_gamma.pdf)

## Contributing

1. Follow the project structure outlined in `AI_README.md`
2. Use functional programming where appropriate
3. Implement rigorous statistical validation
4. Follow PEP 8 style guidelines
5. Add comprehensive documentation

## License

MIT License - see LICENSE file for details.