#!/usr/bin/env python3
"""
CLV Analysis - Main Execution Script

This script runs the complete CLV analysis workflow including:
- Data loading and preprocessing
- Model fitting (BG/NBD and Gamma-Gamma)
- CLV predictions and probability alive calculations
- Visualization and reporting

Usage:
    python src/run_clv_analysis.py
"""

import sys
import os
import numpy as np
# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

# Import all required modules
from src.data_preprocess.data_preprocess_utils import load_retail_data, clean_transaction_data, create_customer_summary
from src.model.lifetimes.models import BGNBDModel, GammaGammaCLVModel, plot_probability_alive_heatmap
from src.model.lifetimes.visualization import generate_html_report

def main():
    """Run the complete CLV analysis workflow."""
    
    # Configuration
    DATA_PATH = '/Users/stephenzhang/Downloads/clv_data/Year_2010_2011.csv'
    
    print("🚀 Starting CLV Analysis...")
    print("=" * 60)
    
    # Step 1: Data Loading and Preprocessing
    print("\n📊 Step 1: Loading and preprocessing data...")
    df_raw = load_retail_data(DATA_PATH)
    print(f"   Raw data shape: {df_raw.shape}")
    
    df_clean = clean_transaction_data(df_raw)
    print(f"   Cleaned data shape: {df_clean.shape}")
    
    summary = create_customer_summary(df_clean)
    print(f"   Customer summary shape: {summary.shape}")
    
    # Step 2: Model Fitting
    print("\n🤖 Step 2: Fitting CLV models...")
    
    # Fit BG/NBD model
    print("   Fitting BG/NBD model...")
    bg_nbd_model = BGNBDModel(penalizer_coef=0.0)
    bg_nbd_model.fit(
        frequency=summary['frequency'],
        recency=summary['recency'], 
        T=summary['T']
    )
    print("   ✅ BG/NBD model fitted successfully!")
    
    # Fit Gamma-Gamma model
    print("   Fitting Gamma-Gamma model...")
    gamma_gamma_model = GammaGammaCLVModel(penalizer_coef=0.0)
    gamma_gamma_model.fit(
        frequency=summary['frequency'],
        monetary_value=summary['monetary_value']
    )
    print("   ✅ Gamma-Gamma model fitted successfully!")
    
    # Step 3: CLV Predictions
    print("\n💰 Step 3: Calculating CLV predictions...")
    
    # Calculate CLV for 30 and 90 days
    clv_30d = gamma_gamma_model.customer_lifetime_value(
        bg_nbd_model=bg_nbd_model,
        frequency=summary['frequency'],
        recency=summary['recency'],
        T=summary['T'],
        monetary_value=summary['monetary_value'],
        time=30, freq='D', discount_rate=0.01
    )
    
    clv_90d = gamma_gamma_model.customer_lifetime_value(
        bg_nbd_model=bg_nbd_model,
        frequency=summary['frequency'],
        recency=summary['recency'],
        T=summary['T'],
        monetary_value=summary['monetary_value'],
        time=90, freq='D', discount_rate=0.01
    )
    
    # Calculate probability alive
    prob_alive = bg_nbd_model.probability_alive(
        frequency=summary['frequency'],
        recency=summary['recency'],
        T=summary['T']
    )
    
    print("   ✅ CLV predictions calculated successfully!")
    
    # Step 4: Results Compilation
    print("\n📈 Step 4: Compiling results...")
    
    # Create comprehensive results dataframe
    summary_with_clv = summary.copy()
    summary_with_clv['clv_30d'] = clv_30d
    summary_with_clv['clv_90d'] = clv_90d
    summary_with_clv['prob_alive'] = prob_alive
    
    # Display key statistics
    print("\n📊 Key Results:")
    print(f"   Total customers analyzed: {len(summary_with_clv):,}")
    print(f"   Mean 30-day CLV: £{clv_30d.mean():.2f}")
    print(f"   Median 30-day CLV: £{np.median(clv_30d):.2f}")
    print(f"   Mean 90-day CLV: £{clv_90d.mean():.2f}")
    print(f"   Median 90-day CLV: £{np.median(clv_90d):.2f}")
    print(f"   Mean probability alive: {prob_alive.mean():.3f}")
    print(f"   Median probability alive: {np.median(prob_alive):.3f}")
    print(f"   High-value customers (>£10k CLV): {(clv_30d > 10000).sum():,}")
    print(f"   At-risk customers (<10% prob alive): {(prob_alive < 0.1).sum():,}")
    
    # Step 5: Save Results and Generate Reports
    print("\n💾 Step 5: Saving results and generating reports...")
    
    # Save complete results
    output_path = 'data/lifetimes/clv_results_complete.csv'
    summary_with_clv.to_csv(output_path, index=True)
    print(f"   ✅ Complete results saved to: {output_path}")
    
    # Generate probability alive heatmap
    print("   Generating probability alive heatmap...")
    
    # Ensure the plots directory exists
    os.makedirs('data/lifetimes/plots', exist_ok=True)
    
    plot_probability_alive_heatmap(
        bg_nbd_model, 
        max_frequency=10, 
        max_recency=30, 
        T=30, 
        save_path='data/lifetimes/plots/probability_alive_heatmap.png'
    )
    print("   ✅ Probability alive heatmap generated!")
    
    # Generate comprehensive visualization report
    print("   Generating HTML visualization report...")
    generate_html_report()
    print("   ✅ HTML visualization report generated!")
    
    # Final summary
    print("\n🎉 CLV Analysis Complete!")
    print("=" * 60)
    print("📁 Output Files:")
    print(f"   1. Complete CLV results: {output_path}")
    print("   2. Probability alive heatmap: data/lifetimes/plots/probability_alive_heatmap.png")
    print("   3. HTML visualization report: data/lifetimes/visualization_report.html")
    print("   4. All plots: data/lifetimes/plots/")
    print("=" * 60)
    
    return summary_with_clv

if __name__ == "__main__":
    results = main()
