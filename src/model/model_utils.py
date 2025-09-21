"""
model_utils.py

Shared utilities for model development and evaluation across different models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple


def plot_histogram(data, column, title, xlabel, bins=50, fname=None, save_dir=None):
    """Create and optionally save a histogram plot."""
    plt.figure(figsize=(8, 5))
    sns.histplot(data[column], bins=bins, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.tight_layout()
    
    if fname and save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, fname))
    
    plt.close()


def plot_scatter(data, x, y, title, xlabel, ylabel, fname=None, save_dir=None):
    """Create and optionally save a scatter plot."""
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=data[x], y=data[y], alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    
    if fname and save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, fname))
    
    plt.close()


def plot_heatmap(data, title, xlabel, ylabel, cmap='YlGnBu', fname=None, save_dir=None):
    """Create and optionally save a heatmap plot."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(data, cmap=cmap)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    
    if fname and save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, fname))
    
    plt.close()


def generate_html_report(plot_paths, report_path, title="Model Analysis Report"):
    """
    Generate an HTML report embedding all saved plots.
    
    Args:
        plot_paths: List of tuples (title, path) for each plot
        report_path: Path to save the HTML report
        title: Title for the HTML report
    """
    import os
    
    with open(report_path, 'w') as f:
        f.write(f'<html><head><title>{title}</title></head><body>\n')
        f.write(f'<h1>{title}</h1>\n')
        
        for plot_title, plot_path in plot_paths:
            f.write(f'<h2>{plot_title}</h2>\n')
            f.write(f'<img src="{os.path.basename(plot_path)}" style="max-width:700px;"><br><br>\n')
        
        f.write('</body></html>\n')


def calculate_model_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate common model evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model for reporting
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {
        'model_name': model_name,
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    return metrics


def cross_validate_model(model, X, y, cv=5, scoring='neg_mean_squared_error'):
    """
    Perform cross-validation on a model.
    
    Args:
        model: Model to validate
        X: Features
        y: Target
        cv: Number of cross-validation folds
        scoring: Scoring metric
        
    Returns:
        Cross-validation results
    """
    from sklearn.model_selection import cross_val_score
    
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    return {
        'cv_scores': cv_scores,
        'mean_score': cv_scores.mean(),
        'std_score': cv_scores.std(),
        'cv_folds': cv
    }
