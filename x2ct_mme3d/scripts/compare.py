#!/usr/bin/env python3
"""
Pulmonary Disease Classification Model Comparison

This script compares a synthetic CT scan backbone model against a baseline CNN model
for pulmonary disease classification, using bootstrapping to assess statistical significance.

The comparison includes:
1. F1 score comparison with confidence intervals
2. ROC-AUC comparison with confidence intervals
3. Statistical significance testing (p-values)
4. Clinical/practical significance assessment
5. Visualization of performance metrics and ROC curves

Expected CSV format:
- CSV with columns: 'uids', 'true', 'pred'
- 'true' column contains binary labels (0 or 1) for pulmonary disease presence
- 'pred' column contains predictions in string format "[0.xxxx]"

Usage:
    python compare.py <baseline_csv> <synthetic_ct_csv> [options]

Options:
    --threshold FLOAT    Classification threshold (default: 0.5)
    --iterations INT     Number of bootstrap iterations (default: 1000)
    --alpha FLOAT        Significance level for confidence intervals (default: 0.05)
    --seed INT           Random seed for reproducibility (default: 42)
    --plot               Generate and display comparison plots
    --stratified         Use stratified bootstrapping for imbalanced datasets
    --baseline-name STR  Custom name for baseline model
    --proposed-name STR  Custom name for proposed model

Example:
    python compare.py baseline-results.csv synthetic-ct-results.csv --plot --stratified
    
To run: 
    python3 compare.py baseline-results.csv b220a09-results.csv --plot
"""

import argparse
import sys
import re
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Union
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from scipy import stats


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare a synthetic CT scan backbone model with a baseline CNN for pulmonary disease classification'
    )
    parser.add_argument('baseline_csv', type=str, help='Path to the baseline CNN model CSV file')
    parser.add_argument('synthetic_ct_csv', type=str, help='Path to the synthetic CT backbone model CSV file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary classification (default: 0.5)')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of bootstrap iterations (default: 1000)')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level for confidence intervals (default: 0.05)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--plot', action='store_true', help='Generate and display comparison plots')
    parser.add_argument('--min-samples', type=int, default=30, help='Minimum recommended sample size (default: 30)')
    parser.add_argument('--stratified', action='store_true', help='Use stratified bootstrapping for imbalanced datasets')
    parser.add_argument('--baseline-name', type=str, default='Baseline CNN', help='Custom name for baseline model')
    parser.add_argument('--proposed-name', type=str, default='Synthetic CT Model', help='Custom name for proposed model')
    parser.add_argument('--detailed', action='store_true', help='Show detailed clinical significance analysis')
    
    return parser.parse_args()


def load_and_validate_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from CSV file and validate its format.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Tuple containing true labels and model predictions
        
    Raises:
        ValueError: If file format is invalid or missing required columns
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check required columns
        required_columns = ['true', 'pred']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in {file_path}: {', '.join(missing_columns)}")
        
        # Extract true labels
        y_true = df['true'].values
        
        # Extract and clean predictions (converting from string format "[0.xxxx]" to float)
        y_pred_raw = df['pred'].values
        y_pred = np.array([float(re.search(r'\[(.*?)\]', str(p)).group(1)) if isinstance(p, str) else p for p in y_pred_raw])
        
        # Check if all values in y_true are binary (0 or 1)
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError(f"True labels in {file_path} must be binary (0 or 1)")
        
        return y_true, y_pred
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Empty file: {file_path}")
    except pd.errors.ParserError:
        raise ValueError(f"Invalid CSV format in file: {file_path}")
    except Exception as e:
        raise ValueError(f"Error loading data from {file_path}: {str(e)}")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate F1 score and ROC-AUC score.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary with F1 and ROC-AUC scores
    """
    # Binary predictions based on threshold
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate metrics with zero_division=0 to avoid warnings
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    # Handle case when all true labels are the same (ROC-AUC undefined)
    if len(np.unique(y_true)) < 2:
        auc = float('nan')
    else:
        auc = roc_auc_score(y_true, y_pred)
    
    return {'f1': f1, 'auc': auc}


def analyze_class_distribution(y_true: np.ndarray) -> Dict[str, Union[int, float, Dict[int, int]]]:
    """
    Analyze the class distribution in the dataset.
    
    Args:
        y_true: True binary labels
        
    Returns:
        Dictionary with class distribution statistics
    """
    n_samples = len(y_true)
    class_counts = {}
    for label in np.unique(y_true):
        class_counts[int(label)] = int(np.sum(y_true == label))
    
    # Calculate class imbalance ratio (majority/minority)
    counts = list(class_counts.values())
    if len(counts) > 1 and min(counts) > 0:
        imbalance_ratio = max(counts) / min(counts)
    else:
        imbalance_ratio = float('inf')
    
    return {
        'n_samples': n_samples,
        'class_counts': class_counts,
        'imbalance_ratio': imbalance_ratio
    }


def bootstrap_metrics(
    y_true: np.ndarray, 
    model1_pred: np.ndarray, 
    model2_pred: np.ndarray, 
    n_iterations: int = 1000,
    threshold: float = 0.5,
    random_seed: int = 42
) -> Dict[str, Dict[str, List[float]]]:
    """
    Perform bootstrapping to calculate distributions of metrics.
    
    Args:
        y_true: True binary labels
        model1_pred: Predictions from model 1
        model2_pred: Predictions from model 2
        n_iterations: Number of bootstrap iterations
        threshold: Threshold for binary classification
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with bootstrapped metrics for both models
    """
    np.random.seed(random_seed)
    n_samples = len(y_true)
    
    # Initialize arrays to store bootstrap results
    model1_f1 = np.zeros(n_iterations)
    model1_auc = np.zeros(n_iterations)
    model2_f1 = np.zeros(n_iterations)
    model2_auc = np.zeros(n_iterations)
    
    # Check if stratified bootstrapping should be used
    class_labels = np.unique(y_true)
    use_stratified = len(class_labels) > 1 and min(np.bincount(y_true.astype(int))) >= 2
    
    # Perform bootstrapping
    for i in range(n_iterations):
        if use_stratified:
            # Stratified bootstrapping (sample from each class separately)
            indices = []
            for label in class_labels:
                label_indices = np.where(y_true == label)[0]
                sampled_indices = np.random.choice(
                    label_indices, 
                    len(label_indices), 
                    replace=True
                )
                indices.extend(sampled_indices)
            indices = np.array(indices)
            np.random.shuffle(indices)  # Shuffle to avoid ordering bias
        else:
            # Regular bootstrapping
            indices = np.random.choice(n_samples, n_samples, replace=True)
        
        # Get bootstrap samples
        bootstrap_true = y_true[indices]
        bootstrap_model1 = model1_pred[indices]
        bootstrap_model2 = model2_pred[indices]
        
        # Calculate metrics for both models
        model1_metrics = calculate_metrics(bootstrap_true, bootstrap_model1, threshold)
        model2_metrics = calculate_metrics(bootstrap_true, bootstrap_model2, threshold)
        
        # Store results
        model1_f1[i] = model1_metrics['f1']
        model1_auc[i] = model1_metrics['auc']
        model2_f1[i] = model2_metrics['f1']
        model2_auc[i] = model2_metrics['auc']
    
    return {
        'model1': {'f1': model1_f1, 'auc': model1_auc},
        'model2': {'f1': model2_f1, 'auc': model2_auc}
    }


def calculate_confidence_intervals(
    bootstrap_results: Dict[str, Dict[str, List[float]]],
    alpha: float = 0.05
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Calculate confidence intervals from bootstrap results.
    
    Args:
        bootstrap_results: Results from bootstrapping
        alpha: Significance level (e.g., 0.05 for 95% confidence)
        
    Returns:
        Dictionary with confidence intervals for each metric and model
    """
    results = {}
    
    for model_name, metrics in bootstrap_results.items():
        results[model_name] = {}
        
        for metric_name, values in metrics.items():
            # Filter out NaN values
            valid_values = np.array([v for v in values if not np.isnan(v)])
            
            if len(valid_values) > 0:
                # Calculate mean and percentile-based confidence intervals
                mean_value = np.mean(valid_values)
                lower_bound = np.percentile(valid_values, 100 * (alpha / 2))
                upper_bound = np.percentile(valid_values, 100 * (1 - alpha / 2))
                
                results[model_name][metric_name] = {
                    'mean': mean_value,
                    'lower': lower_bound,
                    'upper': upper_bound
                }
            else:
                results[model_name][metric_name] = {
                    'mean': float('nan'),
                    'lower': float('nan'),
                    'upper': float('nan')
                }
    
    return results


def calculate_pvalues(
    bootstrap_results: Dict[str, Dict[str, List[float]]]
) -> Dict[str, float]:
    """
    Calculate p-values for statistical significance testing.
    
    Args:
        bootstrap_results: Results from bootstrapping
        
    Returns:
        Dictionary with p-values for each metric
    """
    p_values = {}
    
    for metric_name in ['f1', 'auc']:
        # Get metric values for both models
        model1_values = np.array(bootstrap_results['model1'][metric_name])
        model2_values = np.array(bootstrap_results['model2'][metric_name])
        
        # Remove NaN values
        valid_indices = ~(np.isnan(model1_values) | np.isnan(model2_values))
        model1_valid = model1_values[valid_indices]
        model2_valid = model2_values[valid_indices]
        
        if len(model1_valid) > 0 and len(model2_valid) > 0:
            # Calculate differences for paired test
            differences = model2_valid - model1_valid
            
            # Perform bootstrap hypothesis test (proportion of differences <= 0)
            if np.mean(differences) >= 0:
                p_value = np.mean(differences <= 0)
            else:
                p_value = np.mean(differences >= 0)
            
            # Two-tailed p-value
            p_values[metric_name] = min(p_value, 1 - p_value) * 2
        else:
            p_values[metric_name] = float('nan')
    
    return p_values

def plot_roc_curves(
    y_true: np.ndarray,
    model1_pred: np.ndarray,
    model2_pred: np.ndarray,
    model1_name: str,
    model2_name: str
) -> plt.Figure:
    """
    Generate ROC curves for both models.
    
    Args:
        y_true: True binary labels
        model1_pred: Predictions from model 1
        model2_pred: Predictions from model 2
        model1_name: Name of the first model
        model2_name: Name of the second model
        
    Returns:
        Matplotlib figure with ROC curves
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Calculate ROC curve for model 1
    fpr1, tpr1, _ = roc_curve(y_true, model1_pred)
    roc_auc1 = roc_auc_score(y_true, model1_pred)
    
    # Calculate ROC curve for model 2
    fpr2, tpr2, _ = roc_curve(y_true, model2_pred)
    roc_auc2 = roc_auc_score(y_true, model2_pred)
    
    # Plot ROC curves
    ax.plot(fpr1, tpr1, lw=2, label=f'{model1_name} (AUC = {roc_auc1:.3f})')
    ax.plot(fpr2, tpr2, lw=2, label=f'{model2_name} (AUC = {roc_auc2:.3f})')
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    
    # Set labels and title
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves')
    
    # Add legend and grid
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    return fig


def plot_bootstrap_distributions(
    bootstrap_results: Dict[str, Dict[str, List[float]]],
    model1_name: str,
    model2_name: str,
    dataset_stats: Dict[str, Union[int, float, Dict[int, int]]] = None,
    y_true: np.ndarray = None,
    model1_pred: np.ndarray = None,
    model2_pred: np.ndarray = None
) -> None:
    """
    Generate plots of bootstrap distributions and ROC curves.
    
    Args:
        bootstrap_results: Results from bootstrapping
        model1_name: Name of the first model
        model2_name: Name of the second model
        dataset_stats: Statistics about the dataset (optional)
        y_true: True binary labels (for ROC curves)
        model1_pred: Predictions from model 1 (for ROC curves)
        model2_pred: Predictions from model 2 (for ROC curves)
    """
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Define subplot grid
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # F1 score distribution subplot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(bootstrap_results['model1']['f1'], alpha=0.5, label=model1_name, bins=20)
    ax1.hist(bootstrap_results['model2']['f1'], alpha=0.5, label=model2_name, bins=20)
    ax1.set_title('F1 Score Distribution', fontsize=14)
    ax1.set_xlabel('F1 Score')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # Add vertical lines for means
    model1_f1_mean = np.nanmean(bootstrap_results['model1']['f1'])
    model2_f1_mean = np.nanmean(bootstrap_results['model2']['f1'])
    ax1.axvline(model1_f1_mean, color='blue', linestyle='--', alpha=0.7)
    ax1.axvline(model2_f1_mean, color='orange', linestyle='--', alpha=0.7)
    
    # Add difference annotation
    f1_diff = model2_f1_mean - model1_f1_mean
    ax1.annotate(f'Difference: {f1_diff:.4f}',
                xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
    
    # ROC-AUC distribution subplot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(bootstrap_results['model1']['auc'], alpha=0.5, label=model1_name, bins=20)
    ax2.hist(bootstrap_results['model2']['auc'], alpha=0.5, label=model2_name, bins=20)
    ax2.set_title('ROC-AUC Distribution', fontsize=14)
    ax2.set_xlabel('ROC-AUC')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    # Add vertical lines for means
    model1_auc_mean = np.nanmean(bootstrap_results['model1']['auc'])
    model2_auc_mean = np.nanmean(bootstrap_results['model2']['auc'])
    ax2.axvline(model1_auc_mean, color='blue', linestyle='--', alpha=0.7)
    ax2.axvline(model2_auc_mean, color='orange', linestyle='--', alpha=0.7)
    
    # Add difference annotation
    auc_diff = model2_auc_mean - model1_auc_mean
    ax2.annotate(f'Difference: {auc_diff:.4f}',
                xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
    
    # ROC curves subplot (if true labels and predictions are provided)
    if y_true is not None and model1_pred is not None and model2_pred is not None:
        ax3 = fig.add_subplot(gs[1, :])
        
        # Calculate ROC curve for model 1
        fpr1, tpr1, _ = roc_curve(y_true, model1_pred)
        roc_auc1 = roc_auc_score(y_true, model1_pred)
        
        # Calculate ROC curve for model 2
        fpr2, tpr2, _ = roc_curve(y_true, model2_pred)
        roc_auc2 = roc_auc_score(y_true, model2_pred)
        
        # Plot ROC curves
        ax3.plot(fpr1, tpr1, lw=2, label=f'{model1_name} (AUC = {roc_auc1:.3f})')
        ax3.plot(fpr2, tpr2, lw=2, label=f'{model2_name} (AUC = {roc_auc2:.3f})')
        
        # Plot diagonal line (random classifier)
        ax3.plot([0, 1], [0, 1], 'k--', lw=1)
        
        # Set labels and title
        ax3.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        ax3.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        ax3.set_title('Receiver Operating Characteristic (ROC) Curves for Pulmonary Disease Detection', fontsize=14)
        
        # Add legend and grid
        ax3.legend(loc="lower right")
        ax3.grid(alpha=0.3)
        
        # Set axis limits
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
    
    # Add dataset info as text if available
    if dataset_stats:
        plt.figtext(0.5, 0.01, 
                   f"Test Dataset: {dataset_stats['n_samples']} samples, " +
                   f"Class distribution: {dataset_stats['class_counts']}", 
                   ha="center", fontsize=12, 
                   bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
        
        # Add warning for small sample size
        if dataset_stats['n_samples'] < 30:
            plt.figtext(0.5, 0.99, 
                       "⚠️ WARNING: Small sample size - Results may not be statistically reliable", 
                       ha="center", fontsize=14, color='red',
                       bbox={"facecolor":"yellow", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, top=0.92, hspace=0.3)  # Make room for the dataset info text
    plt.show()


def assess_clinical_significance(
    confidence_intervals: Dict[str, Dict[str, Dict[str, float]]],
    model1_name: str,
    model2_name: str
) -> Dict[str, Dict[str, Union[float, str, bool]]]:
    """
    Assess the clinical significance of model differences for pulmonary disease detection.
    
    Args:
        confidence_intervals: Confidence intervals for metrics
        model1_name: Name of the first model
        model2_name: Name of the second model
        
    Returns:
        Dictionary with clinical significance assessments
    """
    # Calculate differences
    f1_diff = confidence_intervals['model2']['f1']['mean'] - confidence_intervals['model1']['f1']['mean']
    auc_diff = confidence_intervals['model2']['auc']['mean'] - confidence_intervals['model1']['auc']['mean']
    
    # Thresholds for clinical significance
    # These thresholds are examples and should be adjusted based on domain expertise
    f1_threshold = 0.05  # 5% improvement in F1 score is clinically meaningful
    auc_threshold = 0.03  # 3% improvement in AUC is clinically meaningful
    
    # Assess clinical significance
    f1_clinically_significant = abs(f1_diff) >= f1_threshold
    auc_clinically_significant = abs(auc_diff) >= auc_threshold
    
    # Determine clinical impact for each metric
    if f1_diff > 0:
        f1_impact = f"Proposed model may improve detection accuracy by {f1_diff:.1%}"
        f1_better_model = model2_name
    else:
        f1_impact = f"Baseline model performs better in terms of detection accuracy by {abs(f1_diff):.1%}"
        f1_better_model = model1_name
        
    if auc_diff > 0:
        auc_impact = f"Proposed model improves discrimination ability by {auc_diff:.1%}"
        auc_better_model = model2_name
    else:
        auc_impact = f"Baseline model has better discrimination ability by {abs(auc_diff):.1%}"
        auc_better_model = model1_name
    
    # Overall assessment
    if f1_clinically_significant and auc_clinically_significant:
        if f1_diff > 0 and auc_diff > 0:
            overall = "The synthetic CT backbone model shows clinically meaningful improvements in both detection accuracy and discrimination ability."
        elif f1_diff < 0 and auc_diff < 0:
            overall = "The baseline CNN model performs better in clinically meaningful ways for both detection accuracy and discrimination ability."
        else:
            overall = "Models show mixed results with clinically meaningful differences in different aspects of performance."
    elif f1_clinically_significant:
        if f1_diff > 0:
            overall = "The synthetic CT backbone model shows clinically meaningful improvement in detection accuracy, but not in discrimination ability."
        else:
            overall = "The baseline CNN model performs better in detection accuracy in a clinically meaningful way."
    elif auc_clinically_significant:
        if auc_diff > 0:
            overall = "The synthetic CT backbone model shows clinically meaningful improvement in discrimination ability, but not in detection accuracy."
        else:
            overall = "The baseline CNN model has better discrimination ability in a clinically meaningful way."
    else:
        overall = "Neither model shows clinically meaningful performance differences for pulmonary disease detection."
    
    return {
        'f1': {
            'difference': f1_diff,
            'clinically_significant': f1_clinically_significant,
            'threshold': f1_threshold,
            'impact': f1_impact,
            'better_model': f1_better_model
        },
        'auc': {
            'difference': auc_diff,
            'clinically_significant': auc_clinically_significant,
            'threshold': auc_threshold,
            'impact': auc_impact,
            'better_model': auc_better_model
        },
        'overall': overall
    }


def print_results(
    confidence_intervals: Dict[str, Dict[str, Dict[str, float]]],
    p_values: Dict[str, float],
    model1_name: str,
    model2_name: str,
    dataset_stats: Dict[str, Union[int, float, Dict[int, int]]] = None,
    detailed: bool = False
) -> None:
    """
    Print formatted results for pulmonary disease classification model comparison.
    
    Args:
        confidence_intervals: Confidence intervals for metrics
        p_values: P-values for statistical significance
        model1_name: Name of the first model (baseline)
        model2_name: Name of the second model (proposed)
        dataset_stats: Statistics about the dataset (optional)
        detailed: Whether to show detailed clinical significance analysis
    """
    print("\n" + "=" * 100)
    print(f"PULMONARY DISEASE CLASSIFICATION MODEL COMPARISON")
    print(f"Comparing: {model1_name} (baseline) vs {model2_name} (proposed)")
    print("=" * 100)
    
    # Print dataset information if available
    if dataset_stats:
        print("\n📊 Test Dataset Information:")
        print(f"  • Sample size: {dataset_stats['n_samples']} samples")
        
        # Print class distribution
        print("  • Class distribution:")
        for label, count in dataset_stats['class_counts'].items():
            percentage = (count / dataset_stats['n_samples']) * 100
            class_desc = "Non-disease" if label == 0 else "Pulmonary disease"
            print(f"    - Class {label} ({class_desc}): {count} samples ({percentage:.1f}%)")
        
        # Print imbalance ratio
        print(f"  • Class imbalance ratio: {dataset_stats['imbalance_ratio']:.2f}")
        
        # Print sample size warning if needed
        if dataset_stats['n_samples'] < 30:
            print("\n⚠️  WARNING: Small sample size detected (<30 samples)")
            print("  • Results may not be statistically reliable")
            print("  • Confidence intervals may be wide")
            print("  • Consider collecting more data if possible")
        
        if dataset_stats['imbalance_ratio'] > 3:
            print("\n⚠️  WARNING: Significant class imbalance detected")
            print("  • F1 score is recommended for imbalanced datasets")
            print("  • Consider using stratified bootstrapping for more reliable results")
    
    # Calculate differences for summary
    f1_diff = confidence_intervals['model2']['f1']['mean'] - confidence_intervals['model1']['f1']['mean']
    auc_diff = confidence_intervals['model2']['auc']['mean'] - confidence_intervals['model1']['auc']['mean']
    
    # Print summary
    print("\n📋 SUMMARY OF FINDINGS:")
    print(f"  • F1 Score difference: {f1_diff:.4f} ({'+' if f1_diff >= 0 else ''}{f1_diff*100:.1f}%)")
    print(f"  • ROC-AUC difference: {auc_diff:.4f} ({'+' if auc_diff >= 0 else ''}{auc_diff*100:.1f}%)")
    print(f"  • Statistical significance: {'Yes' if (p_values['f1'] < 0.05 or p_values['auc'] < 0.05) else 'No'}")
    
    # Print metrics section
    print("\n📈 DETAILED METRICS:")
    
    # Print F1 scores
    print("\n  F1 Score (Detection Accuracy):")
    print(f"  • {model1_name}: {confidence_intervals['model1']['f1']['mean']:.4f} " +
          f"[95% CI: {confidence_intervals['model1']['f1']['lower']:.4f} to {confidence_intervals['model1']['f1']['upper']:.4f}]")
    print(f"  • {model2_name}: {confidence_intervals['model2']['f1']['mean']:.4f} " +
          f"[95% CI: {confidence_intervals['model2']['f1']['lower']:.4f} to {confidence_intervals['model2']['f1']['upper']:.4f}]")
    print(f"  • Mean difference: {f1_diff:.4f} ({'+' if f1_diff >= 0 else ''}{f1_diff*100:.1f}%)")
    print(f"  • P-value: {p_values['f1']:.4f}" + 
          f" ({'Statistically significant' if p_values['f1'] < 0.05 else 'Not statistically significant'})")
    
    # Print ROC-AUC scores
    print("\n  ROC-AUC Score (Discrimination Ability):")
    print(f"  • {model1_name}: {confidence_intervals['model1']['auc']['mean']:.4f} " +
          f"[95% CI: {confidence_intervals['model1']['auc']['lower']:.4f} to {confidence_intervals['model1']['auc']['upper']:.4f}]")
    print(f"  • {model2_name}: {confidence_intervals['model2']['auc']['mean']:.4f} " +
          f"[95% CI: {confidence_intervals['model2']['auc']['lower']:.4f} to {confidence_intervals['model2']['auc']['upper']:.4f}]")
    print(f"  • Mean difference: {auc_diff:.4f} ({'+' if auc_diff >= 0 else ''}{auc_diff*100:.1f}%)")
    print(f"  • P-value: {p_values['auc']:.4f}" + 
          f" ({'Statistically significant' if p_values['auc'] < 0.05 else 'Not statistically significant'})")
    
    # Assess clinical significance
    clinical_significance = assess_clinical_significance(confidence_intervals, model1_name, model2_name)
    
    # Print statistical interpretation
    print("\n🔍 STATISTICAL INTERPRETATION:")
    # Interpret F1 results
    if p_values['f1'] < 0.05:
        if f1_diff > 0:
            print(f"  • {model2_name} has significantly better F1 score than {model1_name} (+{f1_diff:.4f})")
        else:
            print(f"  • {model1_name} has significantly better F1 score than {model2_name} ({f1_diff:.4f})")
    else:
        print(f"  • No significant difference in F1 scores between models (diff: {f1_diff:.4f})")
    
    # Interpret AUC results
    if p_values['auc'] < 0.05:
        if auc_diff > 0:
            print(f"  • {model2_name} has significantly better ROC-AUC than {model1_name} (+{auc_diff:.4f})")
        else:
            print(f"  • {model1_name} has significantly better ROC-AUC than {model2_name} ({auc_diff:.4f})")
    else:
        print(f"  • No significant difference in ROC-AUC between models (diff: {auc_diff:.4f})")
    
    # Add additional caution for small sample sizes
    if dataset_stats and dataset_stats['n_samples'] < 30:
        print("\n⚠️  NOTE: Due to the small sample size, the statistical power of this comparison is limited.")
        print("    Consider these results as preliminary and interpret with caution.")
    
    # Print clinical significance section
    print("\n🏥 CLINICAL SIGNIFICANCE ASSESSMENT:")
    print(f"  • F1 Score (Detection Accuracy):")
    print(f"    - Difference: {f1_diff:.4f} ({'+' if f1_diff >= 0 else ''}{f1_diff*100:.1f}%)")
    print(f"    - Clinically significant: {'Yes' if clinical_significance['f1']['clinically_significant'] else 'No'}")
    print(f"    - Clinical impact: {clinical_significance['f1']['impact']}")
    
    print(f"\n  • ROC-AUC (Discrimination Ability):")
    print(f"    - Difference: {auc_diff:.4f} ({'+' if auc_diff >= 0 else ''}{auc_diff*100:.1f}%)")
    print(f"    - Clinically significant: {'Yes' if clinical_significance['auc']['clinically_significant'] else 'No'}")
    print(f"    - Clinical impact: {clinical_significance['auc']['impact']}")
    
    print(f"\n  • Overall assessment:")
    print(f"    {clinical_significance['overall']}")
    
    # Add detailed clinical interpretation if requested
    if detailed:
        print("\n🔬 DETAILED CLINICAL INTERPRETATION:")
        print("  • For Pulmonary Disease Detection:")
        
        # Interpret F1 score differences in clinical context
        if f1_diff >= 0.10:
            print("    - The synthetic CT backbone model shows substantial improvement in correctly")
            print("      identifying pulmonary diseases, potentially reducing missed diagnoses.")
        elif f1_diff >= 0.05:
            print("    - The synthetic CT backbone model shows meaningful improvement in detection")
            print("      accuracy, which may benefit clinical decision-making.")
        elif f1_diff > 0:
            print("    - The synthetic CT backbone model shows modest improvement in detection")
            print("      accuracy, though the clinical impact may be limited.")
        elif f1_diff > -0.05:
            print("    - The baseline CNN model performs slightly better for detection accuracy,")
            print("      but the difference may not impact clinical outcomes.")
        else:
            print("    - The baseline CNN model performs notably better for detection accuracy,")
            print("      suggesting it may be more reliable for clinical use.")
        
        # Interpret AUC differences in clinical context
        if auc_diff >= 0.10:
            print("\n    - The synthetic CT backbone model demonstrates substantially better ability")
            print("      to discriminate between disease and non-disease cases, potentially")
            print("      reducing both false positives and false negatives.")
        elif auc_diff >= 0.03:
            print("\n    - The synthetic CT backbone model shows clinically relevant improvement")
            print("      in discrimination ability, which may reduce unnecessary follow-up tests.")
        elif auc_diff > 0:
            print("\n    - The synthetic CT backbone model shows slight improvement in discrimination")
            print("      ability, though the clinical impact may be minimal.")
        elif auc_diff > -0.03:
            print("\n    - The baseline CNN model has slightly better discrimination ability,")
            print("      but the difference may not significantly affect clinical decisions.")
        else:
            print("\n    - The baseline CNN model demonstrates notably better discrimination")
            print("      ability, suggesting it may be more reliable for clinical use.")
        
        # Add practical implementation considerations
        print("\n  • Practical Implementation Considerations:")
        print("    - Computational requirements")
        print("    - Integration with existing clinical workflows")
        print("    - Need for additional training or expertise")
        print("    - Robustness across different patient populations")
    
    print("\n" + "=" * 100)


def main() -> int:
    """Main function for pulmonary disease classification model comparison."""
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        print(f"Loading data from {args.baseline_csv} and {args.synthetic_ct_csv}...")
        
        # Load and validate data
        y_true1, model1_pred = load_and_validate_data(args.baseline_csv)
        y_true2, model2_pred = load_and_validate_data(args.synthetic_ct_csv)
        
        # Check if true labels are compatible
        if len(y_true1) != len(y_true2) or not np.array_equal(y_true1, y_true2):
            print("Warning: True labels in the two files are different or have different lengths.")
            print("Using the first file's labels for the comparison.")
        
        y_true = y_true1
        
        # Analyze dataset statistics
        dataset_stats = analyze_class_distribution(y_true)
        
        # Sample size warning
        min_sample_size = args.min_samples
        if dataset_stats['n_samples'] < min_sample_size:
            print(f"\n⚠️  WARNING: Small sample size detected ({dataset_stats['n_samples']} < {min_sample_size})")
            print("  Statistical results may not be reliable with small samples.")
            print("  Consider collecting more data for more robust conclusions.")
        
        # Class imbalance warning
        if dataset_stats['imbalance_ratio'] > 3:
            print(f"\n⚠️  WARNING: Class imbalance detected (ratio: {dataset_stats['imbalance_ratio']:.2f})")
            print("  F1 score is preferred for imbalanced datasets.")
            if not args.stratified:
                print("  Consider using --stratified option for more reliable results.")
        
        # Calculate direct metrics
        model1_metrics = calculate_metrics(y_true, model1_pred, args.threshold)
        model2_metrics = calculate_metrics(y_true, model2_pred, args.threshold)
        
        print(f"Running bootstrap analysis with {args.iterations} iterations...")
        if args.stratified:
            print("Using stratified bootstrapping for handling class imbalance...")
        
        # Perform bootstrapping
        bootstrap_results = bootstrap_metrics(
            y_true, model1_pred, model2_pred, 
            n_iterations=args.iterations,
            threshold=args.threshold,
            random_seed=args.seed
        )
        
        # Calculate confidence intervals
        confidence_intervals = calculate_confidence_intervals(bootstrap_results, args.alpha)
        
        # Calculate p-values
        p_values = calculate_pvalues(bootstrap_results)
        
        # Use custom model names if provided, otherwise extract from file paths
        model1_name = args.baseline_name if hasattr(args, 'baseline_name') else args.baseline_csv.split('/')[-1].replace('.csv', '')
        model2_name = args.proposed_name if hasattr(args, 'proposed_name') else args.synthetic_ct_csv.split('/')[-1].replace('.csv', '')
        
        # Print results
        print_results(
            confidence_intervals, 
            p_values, 
            model1_name, 
            model2_name,
            dataset_stats,
            args.detailed
        )
        
        # Generate plots if requested
        if args.plot:
            plot_bootstrap_distributions(
                bootstrap_results, 
                model1_name, 
                model2_name, 
                dataset_stats,
                y_true, 
                model1_pred, 
                model2_pred
            )
        
        return 0
    
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
