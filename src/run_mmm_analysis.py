"""
Main execution script for Meridian MMM analysis.

This script provides a command-line interface to run complete
Marketing Mix Modeling analysis using Google's Meridian framework.
"""

import argparse
import sys
import os
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.meridian.mmm_analysis_poc import run_complete_mmm_analysis
from src.model.meridian.visualization import generate_mmm_html_report


def main():
    """Main function to run MMM analysis."""
    parser = argparse.ArgumentParser(
        description="Run Meridian Marketing Mix Modeling analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with sample data
  python src/run_mmm_analysis.py --use-sample-data

  # Run with custom data file
  python src/run_mmm_analysis.py --data-path data/mmm_data.csv

  # Run with custom parameters
  python src/run_mmm_analysis.py --use-sample-data --n-weeks 104 --n-geos 5 --num-samples 2000
        """
    )
    
    # Data options
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to CSV file containing MMM data'
    )
    parser.add_argument(
        '--use-sample-data',
        action='store_true',
        help='Use sample data for analysis'
    )
    parser.add_argument(
        '--n-weeks',
        type=int,
        default=52,
        help='Number of weeks for sample data (default: 52)'
    )
    parser.add_argument(
        '--n-geos',
        type=int,
        default=3,
        help='Number of geographic regions for sample data (default: 3)'
    )
    
    # Model parameters
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='Number of MCMC samples (default: 1000)'
    )
    parser.add_argument(
        '--num-warmup',
        type=int,
        default=500,
        help='Number of warmup samples (default: 500)'
    )
    parser.add_argument(
        '--num-chains',
        type=int,
        default=2,
        help='Number of MCMC chains (default: 2)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Budget optimization
    parser.add_argument(
        '--total-budget',
        type=float,
        default=1000000,
        help='Total budget for optimization (default: 1000000)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/meridian',
        help='Output directory for results (default: data/meridian)'
    )
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate HTML report'
    )
    parser.add_argument(
        '--report-path',
        type=str,
        default='data/meridian/mmm_analysis_report.html',
        help='Path for HTML report (default: data/meridian/mmm_analysis_report.html)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if not args.data_path and not args.use_sample_data:
        print("Error: Either --data-path or --use-sample-data must be specified")
        sys.exit(1)
    
    if args.data_path and not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        sys.exit(1)
    
    # Print configuration
    print("🚀 Meridian MMM Analysis Configuration")
    print("=" * 50)
    print(f"Data source: {'Sample data' if args.use_sample_data else args.data_path}")
    if args.use_sample_data:
        print(f"Sample data: {args.n_weeks} weeks, {args.n_geos} geos")
    print(f"MCMC samples: {args.num_samples}")
    print(f"Warmup samples: {args.num_warmup}")
    print(f"Chains: {args.num_chains}")
    print(f"Random seed: {args.random_seed}")
    print(f"Total budget: ${args.total_budget:,.0f}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 50)
    
    try:
        # Run complete MMM analysis
        results = run_complete_mmm_analysis(
            data_path=args.data_path,
            use_sample_data=args.use_sample_data,
            n_weeks=args.n_weeks,
            n_geos=args.n_geos,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            total_budget=args.total_budget,
            output_dir=args.output_dir
        )
        
        # Generate HTML report if requested
        if args.generate_report:
            print("\n📊 Generating HTML report...")
            generate_mmm_html_report(results, args.report_path)
        
        # Print summary
        print("\n🎉 Analysis completed successfully!")
        print("\n📈 Key Results:")
        print(f"  Model Performance - R²: {results['analysis']['performance']['r_squared']:.3f}")
        print(f"  RMSE: {results['analysis']['performance']['rmse']:.3f}")
        print(f"  MAE: {results['analysis']['performance']['mae']:.3f}")
        
        print(f"\n💰 Budget Optimization Recommendations:")
        for channel, budget in results['budget_optimization']['budget_allocation'].items():
            efficiency = results['budget_optimization']['efficiency'][channel]
            print(f"  {channel}: ${budget:,.0f} (Efficiency: {efficiency:.4f})")
        
        print(f"\n📁 Results saved to: {args.output_dir}")
        if args.generate_report:
            print(f"📄 HTML report: {args.report_path}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
