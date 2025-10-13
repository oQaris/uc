#!/usr/bin/env python3
"""
Analyze solver performance based on instance parameters
Generates visualizations and statistical analysis of solve time dependencies
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse


def load_results(csv_file):
    """Load results from CSV file"""
    df = pd.read_csv(csv_file)
    # Filter only successful solves
    df = df[df['status'].isin(['optimal', 'feasible'])].copy()
    return df


def add_derived_features(df):
    """Add derived features for analysis"""
    # Add dataset type first
    df['dataset'] = df['file_path'].apply(lambda x: x.split('/')[1] if '/' in x else 'unknown')

    # Variable density per constraint
    df['vars_per_constraint'] = df['approx_total_vars'] / df['approx_constraints']

    # Binary variable ratio
    df['binary_ratio'] = df['approx_binary_vars'] / df['approx_total_vars']

    # Problem size indicator
    df['problem_size'] = df['approx_total_vars'] * df['approx_constraints']

    # Reserve ratio
    df['reserve_ratio'] = df['total_reserves'] / df['peak_demand']

    # Demand utilization
    df['demand_utilization'] = df['avg_demand'] / df['peak_demand']

    # Generator density
    df['gens_per_period'] = df['n_thermal_gens'] / df['time_periods']

    # Startup complexity
    df['startup_per_gen'] = df['total_startup_categories'] / df['n_thermal_gens']

    # PWL complexity
    df['pwl_per_gen'] = df['total_pwl_points'] / df['n_thermal_gens']

    # Must run ratio
    df['must_run_ratio'] = df['n_must_run'] / df['n_thermal_gens']

    return df


def correlation_analysis(df):
    """Analyze correlation between parameters and solve time"""
    # Select numeric columns for correlation
    numeric_cols = [
        'solve_time', 'time_periods', 'n_thermal_gens', 'n_renewable_gens',
        'n_must_run', 'total_startup_categories', 'total_pwl_points',
        'approx_binary_vars', 'approx_continuous_vars', 'approx_total_vars',
        'approx_constraints', 'peak_demand', 'avg_demand', 'total_reserves',
        'vars_per_constraint', 'binary_ratio', 'problem_size', 'reserve_ratio',
        'demand_utilization', 'gens_per_period', 'startup_per_gen',
        'pwl_per_gen', 'must_run_ratio'
    ]

    # Remove columns that might not exist or have missing values
    available_cols = [col for col in numeric_cols if col in df.columns]

    corr_with_time = df[available_cols].corr()['solve_time'].sort_values(ascending=False)

    print("="*80)
    print("CORRELATION WITH SOLVE TIME")
    print("="*80)
    print(corr_with_time)
    print()

    return corr_with_time


def create_visualizations(df, output_dir='analysis_plots'):
    """Create various visualizations"""
    Path(output_dir).mkdir(exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # 1. Correlation heatmap
    print("Creating correlation heatmap...")
    numeric_cols = [
        'solve_time', 'n_thermal_gens', 'total_startup_categories',
        'total_pwl_points', 'approx_binary_vars', 'approx_total_vars',
        'approx_constraints', 'peak_demand', 'total_reserves',
        'binary_ratio', 'reserve_ratio', 'must_run_ratio'
    ]
    available_cols = [col for col in numeric_cols if col in df.columns]

    plt.figure(figsize=(14, 12))
    corr_matrix = df[available_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Instance Parameters and Solve Time', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Solve time vs key parameters
    print("Creating scatter plots...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Solve Time vs Key Parameters', fontsize=16, y=1.00)

    params = [
        ('approx_total_vars', 'Total Variables'),
        ('approx_binary_vars', 'Binary Variables'),
        ('n_thermal_gens', 'Number of Thermal Generators'),
        ('total_startup_categories', 'Total Startup Categories'),
        ('total_reserves', 'Total Reserves'),
        ('peak_demand', 'Peak Demand')
    ]

    for idx, (param, label) in enumerate(params):
        ax = axes[idx // 3, idx % 3]
        for dataset in df['dataset'].unique():
            mask = df['dataset'] == dataset
            ax.scatter(df[mask][param], df[mask]['solve_time'],
                      label=dataset, alpha=0.6, s=50)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Solve Time (s)', fontsize=11)
        ax.set_title(f'Solve Time vs {label}', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/solve_time_vs_parameters.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Problem size analysis
    print("Creating problem size analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Log scale plot
    for dataset in df['dataset'].unique():
        mask = df['dataset'] == dataset
        axes[0].scatter(df[mask]['problem_size'], df[mask]['solve_time'],
                       label=dataset, alpha=0.6, s=50)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Problem Size (vars × constraints)', fontsize=12)
    axes[0].set_ylabel('Solve Time (s)', fontsize=12)
    axes[0].set_title('Solve Time vs Problem Size (log-log)', fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Binary ratio impact
    scatter = axes[1].scatter(df['binary_ratio'], df['solve_time'],
                             c=df['approx_total_vars'], cmap='viridis',
                             alpha=0.6, s=50)
    axes[1].set_xlabel('Binary Variable Ratio', fontsize=12)
    axes[1].set_ylabel('Solve Time (s)', fontsize=12)
    axes[1].set_title('Solve Time vs Binary Ratio (colored by total vars)', fontsize=13)
    plt.colorbar(scatter, ax=axes[1], label='Total Variables')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/problem_size_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Reserve impact analysis
    print("Creating reserve impact analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Reserves vs solve time
    for dataset in df['dataset'].unique():
        mask = df['dataset'] == dataset
        axes[0].scatter(df[mask]['total_reserves'], df[mask]['solve_time'],
                       label=dataset, alpha=0.6, s=50)
    axes[0].set_xlabel('Total Reserves', fontsize=12)
    axes[0].set_ylabel('Solve Time (s)', fontsize=12)
    axes[0].set_title('Solve Time vs Total Reserves', fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Reserve ratio vs solve time
    for dataset in df['dataset'].unique():
        mask = df['dataset'] == dataset
        axes[1].scatter(df[mask]['reserve_ratio'], df[mask]['solve_time'],
                       label=dataset, alpha=0.6, s=50)
    axes[1].set_xlabel('Reserve Ratio (reserves/peak_demand)', fontsize=12)
    axes[1].set_ylabel('Solve Time (s)', fontsize=12)
    axes[1].set_title('Solve Time vs Reserve Ratio', fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/reserve_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Dataset comparison
    print("Creating dataset comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Box plot of solve times by dataset
    df.boxplot(column='solve_time', by='dataset', ax=axes[0])
    axes[0].set_xlabel('Dataset', fontsize=12)
    axes[0].set_ylabel('Solve Time (s)', fontsize=12)
    axes[0].set_title('Solve Time Distribution by Dataset', fontsize=13)
    plt.sca(axes[0])
    plt.xticks(rotation=45)

    # Average metrics by dataset
    dataset_stats = df.groupby('dataset').agg({
        'solve_time': 'mean',
        'approx_total_vars': 'mean',
        'n_thermal_gens': 'mean'
    }).reset_index()

    x = np.arange(len(dataset_stats))
    width = 0.25

    axes[1].bar(x - width, dataset_stats['solve_time'], width, label='Avg Solve Time (s)', alpha=0.8)
    axes[1].bar(x, dataset_stats['approx_total_vars']/1000, width, label='Avg Total Vars (×1000)', alpha=0.8)
    axes[1].bar(x + width, dataset_stats['n_thermal_gens'], width, label='Avg Thermal Gens', alpha=0.8)

    axes[1].set_xlabel('Dataset', fontsize=12)
    axes[1].set_ylabel('Value', fontsize=12)
    axes[1].set_title('Average Metrics by Dataset', fontsize=13)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(dataset_stats['dataset'], rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/dataset_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Generator complexity analysis
    print("Creating generator complexity analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Startup categories impact
    for dataset in df['dataset'].unique():
        mask = df['dataset'] == dataset
        axes[0, 0].scatter(df[mask]['startup_per_gen'], df[mask]['solve_time'],
                          label=dataset, alpha=0.6, s=50)
    axes[0, 0].set_xlabel('Startup Categories per Generator', fontsize=11)
    axes[0, 0].set_ylabel('Solve Time (s)', fontsize=11)
    axes[0, 0].set_title('Solve Time vs Startup Complexity', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # PWL points impact
    for dataset in df['dataset'].unique():
        mask = df['dataset'] == dataset
        axes[0, 1].scatter(df[mask]['pwl_per_gen'], df[mask]['solve_time'],
                          label=dataset, alpha=0.6, s=50)
    axes[0, 1].set_xlabel('PWL Points per Generator', fontsize=11)
    axes[0, 1].set_ylabel('Solve Time (s)', fontsize=11)
    axes[0, 1].set_title('Solve Time vs PWL Complexity', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Must-run ratio impact
    for dataset in df['dataset'].unique():
        mask = df['dataset'] == dataset
        axes[1, 0].scatter(df[mask]['must_run_ratio'], df[mask]['solve_time'],
                          label=dataset, alpha=0.6, s=50)
    axes[1, 0].set_xlabel('Must-Run Generator Ratio', fontsize=11)
    axes[1, 0].set_ylabel('Solve Time (s)', fontsize=11)
    axes[1, 0].set_title('Solve Time vs Must-Run Ratio', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Generators per period
    for dataset in df['dataset'].unique():
        mask = df['dataset'] == dataset
        axes[1, 1].scatter(df[mask]['gens_per_period'], df[mask]['solve_time'],
                          label=dataset, alpha=0.6, s=50)
    axes[1, 1].set_xlabel('Generators per Time Period', fontsize=11)
    axes[1, 1].set_ylabel('Solve Time (s)', fontsize=11)
    axes[1, 1].set_title('Solve Time vs Generator Density', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/generator_complexity.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nAll plots saved to '{output_dir}/' directory")


def statistical_summary(df):
    """Print statistical summary"""
    print("="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)

    print("\nDataset Statistics:")
    print(df.groupby('dataset').agg({
        'instance': 'count',
        'solve_time': ['min', 'max', 'mean', 'std'],
        'approx_total_vars': 'mean',
        'n_thermal_gens': 'mean',
        'peak_demand': 'mean'
    }).round(2))

    print("\n\nSolve Time Statistics:")
    print(df['solve_time'].describe())

    print("\n\nTop 10 Slowest Instances:")
    slowest = df.nlargest(10, 'solve_time')[['instance', 'solve_time', 'approx_total_vars',
                                               'n_thermal_gens', 'total_reserves', 'dataset']]
    print(slowest.to_string(index=False))

    print("\n\nTop 10 Fastest Instances:")
    fastest = df.nsmallest(10, 'solve_time')[['instance', 'solve_time', 'approx_total_vars',
                                                'n_thermal_gens', 'total_reserves', 'dataset']]
    print(fastest.to_string(index=False))


def regression_analysis(df):
    """Perform simple regression analysis"""
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    print("\n" + "="*80)
    print("REGRESSION ANALYSIS")
    print("="*80)

    # Select features for regression
    feature_cols = [
        'n_thermal_gens', 'total_startup_categories', 'total_pwl_points',
        'approx_binary_vars', 'approx_total_vars', 'approx_constraints',
        'total_reserves', 'peak_demand', 'binary_ratio', 'must_run_ratio'
    ]

    # Remove any missing values
    df_clean = df[feature_cols + ['solve_time']].dropna()

    X = df_clean[feature_cols]
    y = df_clean['solve_time']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Linear Regression
    lr = LinearRegression()
    lr_scores = cross_val_score(lr, X_scaled, y, cv=5, scoring='r2')
    lr.fit(X_scaled, y)

    print("\nLinear Regression:")
    print(f"  R^2 Score (CV): {lr_scores.mean():.3f} (+/- {lr_scores.std():.3f})")
    print("\n  Feature Importance (coefficients):")
    coef_df = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': lr.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    print(coef_df.to_string(index=False))

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
    rf.fit(X, y)

    print("\n\nRandom Forest Regression:")
    print(f"  R^2 Score (CV): {rf_scores.mean():.3f} (+/- {rf_scores.std():.3f})")
    print("\n  Feature Importance:")
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(importance_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Analyze solver performance')
    parser.add_argument('--input', type=str, default='server-solvers/res_parallel.csv',
                       help='Input CSV file with results')
    parser.add_argument('--output-dir', type=str, default='analysis_plots',
                       help='Directory for output plots')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--no-regression', action='store_true',
                       help='Skip regression analysis')

    args = parser.parse_args()

    print("Loading results...")
    df = load_results(args.input)
    print(f"Loaded {len(df)} successful instances\n")

    print("Adding derived features...")
    df = add_derived_features(df)

    # Correlation analysis
    correlation_analysis(df)

    # Statistical summary
    statistical_summary(df)

    # Regression analysis
    if not args.no_regression:
        try:
            regression_analysis(df)
        except ImportError:
            print("\nWarning: sklearn not installed, skipping regression analysis")
            print("Install with: pip install scikit-learn")

    # Create visualizations
    if not args.no_plots:
        create_visualizations(df, args.output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
