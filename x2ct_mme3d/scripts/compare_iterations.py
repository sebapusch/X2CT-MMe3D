import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_metrics(baseline_csv, proposed_csv):
    """
    Loads metrics from the provided CSV files.
    """
    try:
        baseline_df = pd.read_csv(baseline_csv)
        proposed_df = pd.read_csv(proposed_csv)

        # Define metric columns to be used for comparison (accuracy removed)
        metric_cols = ['precision', 'recall', 'f1', 'auc']

        # Verify that all metric columns exist in both dataframes
        if not all(col in baseline_df.columns for col in metric_cols) or \
                not all(col in proposed_df.columns for col in metric_cols):
            raise ValueError("One of the CSV files is missing required metric columns (precision, recall, f1, auc).")

        return baseline_df[metric_cols], proposed_df[metric_cols]

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: {e.filename} not found.")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def perform_bootstrap(model1_metrics, model2_metrics, n_iterations=1000, random_seed=42):
    """
    Performs bootstrapping on the metrics of two models.
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    bootstrap_results = {metric: [] for metric in model1_metrics.columns}

    n_samples_model1 = len(model1_metrics)
    n_samples_model2 = len(model2_metrics)

    for _ in range(n_iterations):
        # Resample with replacement from the iterations for each model
        resampled_model1 = model1_metrics.sample(n=n_samples_model1, replace=True)
        resampled_model2 = model2_metrics.sample(n=n_samples_model2, replace=True)

        # Calculate the mean of each metric for the resampled data
        mean_model1 = resampled_model1.mean()
        mean_model2 = resampled_model2.mean()

        # Calculate the difference in means and store it
        for metric in model1_metrics.columns:
            diff = mean_model2[metric] - mean_model1[metric]
            bootstrap_results[metric].append(diff)

    return bootstrap_results


def calculate_confidence_intervals(bootstrap_results, alpha=0.05):
    """
    Calculates confidence intervals for the bootstrapped differences.
    """
    confidence_intervals = {}
    for metric, values in bootstrap_results.items():
        lower_percentile = (alpha / 2.0) * 100
        upper_percentile = (1 - alpha / 2.0) * 100
        lower_bound = np.percentile(values, lower_percentile)
        upper_bound = np.percentile(values, upper_percentile)
        confidence_intervals[metric] = (lower_bound, upper_bound)
    return confidence_intervals


def calculate_pvalues(bootstrap_results):
    """
    Calculates p-values from the bootstrapped differences.
    This is the probability of observing a difference as extreme as the one
    observed, assuming the null hypothesis (no difference) is true.
    """
    p_values = {}
    for metric, values in bootstrap_results.items():
        # Calculate the proportion of bootstrapped differences that are less than 0
        p_value = np.mean(np.array(values) < 0)
        # Two-tailed p-value
        p_values[metric] = 2 * min(p_value, 1 - p_value)
    return p_values


def print_results(confidence_intervals, p_values, model1_name, model2_name, model1_metrics, model2_metrics):
    """
    Prints the comparison results in a formatted table.
    """
    print(f"\nComparing {model1_name} (Model 1) and {model2_name} (Model 2)")
    print("-" * 98)
    print(
        f"{'Metric':<12} | {'Model 1 Mean':<15} | {'Model 2 Mean':<15} | {'Mean Difference':<18} | {'95% CI':<16} | {'p-value':<10}")
    print("-" * 98)

    for metric in confidence_intervals:
        mean_model1 = model1_metrics[metric].mean()
        mean_model2 = model2_metrics[metric].mean()
        mean_diff = mean_model2 - mean_model1

        ci = confidence_intervals[metric]
        p_val = p_values[metric]

        print(
            f"{metric:<12} | {mean_model1:<15.4f} | {mean_model2:<15.4f} | {mean_diff:<+18.4f} | ({ci[0]:.4f}, {ci[1]:.4f}) | {p_val:<10.4f}")
    print("-" * 98)


def plot_bootstrap_distributions(bootstrap_results, model1_name, model2_name):
    """
    Plots the distributions of the bootstrapped metric differences.
    """
    # Use a cleaner plot style
    sns.set_style("whitegrid")

    n_metrics = len(bootstrap_results)
    # Arrange plots in a 2x2 grid
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10), constrained_layout=True)
    axes = axes.flatten()

    # Add a main title for the entire figure
    fig.suptitle(f"Bootstrap Distributions of Metric Differences ({model2_name} - {model1_name})", fontsize=16, y=1.02)

    for i, (metric, values) in enumerate(bootstrap_results.items()):
        sns.histplot(values, kde=True, ax=axes[i], color="skyblue", edgecolor="black")

        # Calculate confidence interval
        lower_bound, upper_bound = np.percentile(values, [2.5, 97.5])

        # Add lines for CI and zero difference
        axes[i].axvline(lower_bound, color='red', linestyle='--', label='95% CI')
        axes[i].axvline(upper_bound, color='red', linestyle='--')
        axes[i].axvline(0, color='black', linestyle='-', label='No difference')

        # Set cleaner titles and labels
        axes[i].set_title(f"{metric.upper()}", fontsize=14)
        axes[i].set_xlabel(f"Difference in {metric.capitalize()}")
        axes[i].set_ylabel("Frequency")
        axes[i].legend()

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.show()


def main():
    """Main function to run the comparison."""
    parser = argparse.ArgumentParser(
        description="Compare two models based on iteration results using bootstrapping.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("baseline_csv", help="CSV file with metrics for the baseline model.")
    parser.add_argument("proposed_csv", help="CSV file with metrics for the proposed model.")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of bootstrap iterations (default: 1000).")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level for confidence intervals (default: 0.05).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--plot", action="store_true", help="Generate and display comparison plots.")
    parser.add_argument("--baseline-name", type=str, default="Baseline", help="Custom name for the baseline model.")
    parser.add_argument("--proposed-name", type=str, default="Proposed", help="Custom name for the proposed model.")

    args = parser.parse_args()

    try:
        # Load data
        model1_metrics, model2_metrics = load_metrics(args.baseline_csv, args.proposed_csv)

        # Perform bootstrapping
        bootstrap_results = perform_bootstrap(
            model1_metrics, model2_metrics,
            n_iterations=args.iterations,
            random_seed=args.seed
        )

        # Calculate confidence intervals
        confidence_intervals = calculate_confidence_intervals(bootstrap_results, args.alpha)

        # Calculate p-values
        p_values = calculate_pvalues(bootstrap_results)

        # Use custom model names if provided
        model1_name = args.baseline_name
        model2_name = args.proposed_name

        # Print results
        print_results(
            confidence_intervals,
            p_values,
            model1_name,
            model2_name,
            model1_metrics,
            model2_metrics
        )

        # Generate plots if requested
        if args.plot:
            plot_bootstrap_distributions(
                bootstrap_results,
                model1_name,
                model2_name
            )

        return 0

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())