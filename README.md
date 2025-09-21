# MMM-CLV: Marketing Mix Modeling & Customer Lifetime Value Analysis

A comprehensive framework for Marketing Mix Modeling (MMM) using Google's Meridian and Customer Lifetime Value (CLV) analysis using Lifetimes (BG/NBD + Gamma-Gamma) models.

## Project Structure

```
mmm-clv/
├── src/                          # Production-ready code
│   ├── main.ipynb               # Main CLV execution notebook
│   ├── meridian_mmm_analysis.ipynb  # Meridian MMM analysis notebook
│   ├── data_preprocess/         # Data preprocessing modules
│   │   ├── data_preprocess_utils.py  # Shared utilities
│   │   ├── lifetimes/           # Lifetimes-specific preprocessing
│   │   └── meridian/            # Meridian MMM preprocessing
│   └── model/                   # Core model algorithms
│       ├── model_utils.py       # Shared model utilities
│       ├── lifetimes/           # Lifetimes CLV model implementation
│       │   ├── models.py        # BG/NBD and Gamma-Gamma models
│       │   ├── clv_analysis_poc.py  # CLV analysis pipeline
│       │   └── visualization.py # CLV visualization tools
│       └── meridian/            # Meridian MMM model implementation
│           ├── meridian_model.py    # Meridian MMM model
│           ├── mmm_analysis_poc.py  # MMM analysis pipeline
│           └── visualization.py     # MMM visualization tools
├── data/                        # Data storage
│   ├── lifetimes/              # Lifetimes CLV model data
│   └── meridian/               # Meridian MMM model data
├── notebooks/                   # Development and testing
├── test/                        # Unit tests
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Features

### Lifetimes Model (CLV Analysis)
- **BG/NBD Model**: Predicts customer transaction frequency
- **Gamma-Gamma Model**: Estimates customer monetary value
- **Probability Alive**: Calculates customer activity probability
- **Visualization**: Comprehensive plots and HTML reports

### Meridian Model (MMM Analysis)
- **Bayesian Causal Inference**: Advanced statistical methods to measure marketing impact
- **Media Attribution**: Quantify the contribution of each media channel
- **Budget Optimization**: Recommend optimal budget allocation across channels
- **Saturation Effects**: Model diminishing returns of media spend
- **Adstock Effects**: Account for delayed impact of advertising
- **Time Series Analysis**: Analyze media contribution over time
- **Prior-Posterior Analysis**: Bayesian model diagnostics


## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv marketing_env
source marketing_env/bin/activate  # On Windows: marketing_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Meridian from GitHub
pip install git+https://github.com/google/meridian.git

# Verify setup (optional)
python -c "import sys; sys.path.insert(0, '.'); from src.model.lifetimes.models import BGNBDModel; print('✅ CLV setup complete!')"
python -c "import sys; sys.path.insert(0, '.'); from src.model.meridian.meridian_model import MeridianMMMModel; print('✅ MMM setup complete!')"
```

### 2. Prepare Data

#### For CLV Analysis
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

#### For MMM Analysis
Place your marketing mix data CSV file in `data/meridian/` directory.

Expected CSV columns:
- `date`: Date column
- `geo`: Geographic identifier
- `{channel}_spend`: Media spend for each channel (e.g., `TV_spend`, `Digital_spend`)
- `sales`: Target variable
- Control variables (e.g., `price`, `promotion`, `seasonality`)

### 3. Run Analysis

#### CLV Analysis (Customer Lifetime Value)

**Option A: Complete Analysis Script (Recommended)**
```bash
# Run complete CLV analysis with all outputs
python src/run_clv_analysis.py
```

**Option B: Jupyter Notebook (Interactive)**
```bash
jupyter notebook src/main.ipynb
```

**Option C: Individual Components**
```bash
# Lifetimes model analysis only
python src/model/lifetimes/clv_analysis_poc.py

# Generate visualizations only
python src/model/lifetimes/visualization.py
```

#### MMM Analysis (Marketing Mix Modeling)

**Option A: Complete Analysis Script (Recommended)**
```bash
# Run complete MMM analysis with all outputs
python src/run_mmm_analysis.py --use-sample-data

# Or with your own data
python src/run_mmm_analysis.py --data-path data/meridian/your_data.csv
```

**Option B: Jupyter Notebook (Interactive)**
```bash
jupyter notebook src/meridian_mmm_analysis.ipynb
```

**Option C: Individual Components**
```bash
# Meridian MMM analysis only
python src/model/meridian/mmm_analysis_poc.py
```

## Model-Specific Usage

### Lifetimes Model (CLV)

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

### Meridian Model (MMM)

```python
from src.model.meridian.meridian_model import MeridianMMMModel
from src.data_preprocess.meridian.data_preprocess_utils import create_sample_mmm_data

# Generate or load MMM data
data = create_sample_mmm_data(n_weeks=52, n_geos=3)

# Initialize and fit model
model = MeridianMMMModel(media_channels=['TV', 'Digital', 'Radio'])
model.fit(media_data, control_data, target_data)

# Get media attribution
attribution = model.get_media_attribution(media_data, control_data)

# Optimize budget
budget_opt = model.optimize_budget(media_data, total_budget=1000000)
```


## Output Files

After running the analysis, you'll find the following outputs:

### CLV Analysis Outputs

#### Data Files
- `data/lifetimes/clv_results_complete.csv`: Complete CLV analysis results with predictions
- `data/lifetimes/clv_results_sample.csv`: Sample results from previous runs

#### Visualizations
- `data/lifetimes/plots/`: Directory containing all generated plots
  - `probability_alive_heatmap.png`: Customer activity probability heatmap
  - `clv_*_hist.png`: Distribution histograms for CLV metrics
  - `rfm_*_hist.png`: RFM analysis plots
  - `*_scatter.png`: Scatter plots for relationship analysis

#### Reports
- `src/model/lifetimes/visualization_report.html`: Comprehensive HTML report with all visualizations and analysis

### MMM Analysis Outputs

#### Data Files
- `data/meridian/mmm_predictions.csv`: Model predictions with confidence intervals
- `data/meridian/mmm_attribution.csv`: Media attribution results by channel and time period
- `data/meridian/mmm_performance.csv`: Model performance metrics

#### Visualizations
- `data/meridian/plots/`: Directory containing all generated plots
  - `media_attribution.png`: Media attribution analysis
  - `model_performance.png`: Model performance diagnostics
  - `budget_optimization.png`: Budget optimization recommendations
  - `media_response_curves.png`: Media saturation curves
  - `media_contribution_timeseries.png`: Time series media contribution
  - `media_target_curves.png`: Media spend vs target relationships
  - `prior_posterior.png`: Bayesian prior-posterior analysis

#### Reports
- `data/meridian/mmm_analysis_report.html`: Comprehensive HTML report with all MMM analysis results

## Visualization and Reporting

The framework generates comprehensive visualizations including:

### CLV Analysis
- **Raw Data Analysis**: Distribution plots for quantities, prices
- **RFM Analysis**: Recency, Frequency, Monetary value distributions
- **CLV Results**: Model predictions and customer segments
- **Probability Alive Heatmap**: Customer activity probability visualization

### MMM Analysis
- **Media Attribution**: Channel contribution analysis and time series
- **Model Performance**: Actual vs predicted, residuals analysis
- **Budget Optimization**: Current vs recommended allocation
- **Media Response Curves**: Saturation effects visualization
- **Prior-Posterior Analysis**: Bayesian model diagnostics

### HTML Reports
- **Comprehensive Analysis Reports**: Professional HTML reports with embedded visualizations
- **Executive Summaries**: Key insights and recommendations
- **Technical Details**: Model performance metrics and statistical validation

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

### Core Dependencies
- pandas: Data manipulation
- numpy: Numerical computing
- matplotlib/seaborn: Visualization
- scikit-learn: Machine learning utilities
- jupyter: Notebook environment

### CLV Analysis Dependencies
- lifetimes: BG/NBD and Gamma-Gamma models

### MMM Analysis Dependencies
- google-meridian: Google's Meridian MMM framework
- jax/jaxlib: High-performance machine learning
- numpyro: Probabilistic programming
- arviz: Bayesian analysis visualization
- scipy: Scientific computing
- tensorflow-probability: Probabilistic modeling

## References

### CLV Analysis
- [Lifetimes Documentation](https://github.com/CamDavidsonPilon/lifetimes)
- [BG/NBD Model Paper](https://www.brucehardie.com/papers/018/fader_et_al_mksc_05.pdf)
- [Gamma-Gamma Model Paper](https://www.brucehardie.com/notes/025/gamma_gamma.pdf)

### MMM Analysis
- [Google Meridian GitHub Repository](https://github.com/google/meridian)
- [Meridian Documentation](https://github.com/google/meridian/blob/main/README.md)
- [Meridian Getting Started Tutorial](https://github.com/google/meridian/blob/main/demo/Meridian_Getting_Started.ipynb)

## Contributing

1. Follow the project structure outlined in `AI_README.md`
2. Use functional programming where appropriate
3. Implement rigorous statistical validation
4. Follow PEP 8 style guidelines
5. Add comprehensive documentation

## License

MIT License - see LICENSE file for details.