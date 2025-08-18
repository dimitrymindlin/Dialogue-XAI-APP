"""
4_get_raw_overview.py

This script loads filtered experiment data from results/{PROLIFIC_FOLDER_NAME}/3_filtered/ 
and creates basic distribution plots and sanity checks for key metrics.

Expected Inputs:
- results/{PROLIFIC_FOLDER_NAME}/3_filtered/users_filtered.csv

Outputs:
- results/{PROLIFIC_FOLDER_NAME}/4_overview/score_distributions.png
- results/{PROLIFIC_FOLDER_NAME}/4_overview/time_distributions.png
- Console output with basic statistics and sanity checks
"""

from experiment_analysis.analysis_config import RESULTS_DIR
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FILTERED_DIR = os.path.join(RESULTS_DIR, "3_filtered")
OVERVIEW_DIR = os.path.join(RESULTS_DIR, "4_overview")
os.makedirs(OVERVIEW_DIR, exist_ok=True)

# Load filtered data
print(f"Loading filtered data from {FILTERED_DIR}")
users_filtered = pd.read_csv(os.path.join(FILTERED_DIR, "users_filtered.csv"))

print(f"Loaded {len(users_filtered)} users for analysis")
print(f"Available columns: {', '.join(users_filtered.columns)}")

# Define key metrics for analysis
score_columns = ['intro_score', 'learning_score', 'final_score']
time_columns = ['total_exp_time', 'total_learning_time']

# Check which columns exist
available_score_cols = [col for col in score_columns if col in users_filtered.columns]
available_time_cols = [col for col in time_columns if col in users_filtered.columns]

print(f"\nAvailable score columns: {available_score_cols}")
print(f"Available time columns: {available_time_cols}")

# Basic sanity checks
print("\n" + "="*50)
print("SANITY CHECKS")
print("="*50)

# Check score ranges (should be 0-10)
for col in available_score_cols:
    scores = users_filtered[col].dropna()
    min_score = scores.min()
    max_score = scores.max()
    out_of_range = len(scores[(scores < 0) | (scores > 10)])
    print(f"{col}: range {min_score:.1f}-{max_score:.1f}, {out_of_range} values out of 0-10 range")

# Check for negative times
for col in available_time_cols:
    times = users_filtered[col].dropna()
    negative_times = len(times[times < 0])
    print(f"{col}: {negative_times} negative values")

# Count missing values
print(f"\nMissing values:")
for col in available_score_cols + available_time_cols:
    missing = users_filtered[col].isna().sum()
    print(f"{col}: {missing} missing ({missing/len(users_filtered)*100:.1f}%)")

# Basic progression check
if 'intro_score' in available_score_cols and 'final_score' in available_score_cols:
    intro_final_pairs = users_filtered[['intro_score', 'final_score']].dropna()
    improved = len(intro_final_pairs[intro_final_pairs['final_score'] > intro_final_pairs['intro_score']])
    same = len(intro_final_pairs[intro_final_pairs['final_score'] == intro_final_pairs['intro_score']])
    declined = len(intro_final_pairs[intro_final_pairs['final_score'] < intro_final_pairs['intro_score']])
    print(f"\nScore progression (intro -> final): {improved} improved, {same} same, {declined} declined")

# Create score distribution plots (boxplots)
if available_score_cols:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Prepare data for boxplot
    score_data = []
    score_labels = []
    
    for col in available_score_cols:
        data = users_filtered[col].dropna()
        score_data.append(data)
        score_labels.append(col.replace("_", " ").title())
    
    # Create boxplot
    box_plot = ax.boxplot(score_data, labels=score_labels, patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors[:len(score_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title('Score Distributions')
    ax.set_ylabel('Score')
    ax.set_ylim(-0.5, 10.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    score_plot_path = os.path.join(OVERVIEW_DIR, "score_distributions.png")
    plt.savefig(score_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved score distributions to: {score_plot_path}")

# Create time distribution plots
if available_time_cols:
    fig, axes = plt.subplots(1, len(available_time_cols), figsize=(5*len(available_time_cols), 4))
    if len(available_time_cols) == 1:
        axes = [axes]
    
    for i, col in enumerate(available_time_cols):
        data = users_filtered[col].dropna()
        axes[i].hist(data, bins=20, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{col.replace("_", " ").title()}\n(n={len(data)})')
        axes[i].set_xlabel('Time (minutes)')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    time_plot_path = os.path.join(OVERVIEW_DIR, "time_distributions.png")
    plt.savefig(time_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved time distributions to: {time_plot_path}")

# Print summary statistics
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)

all_metrics = available_score_cols + available_time_cols
if all_metrics:
    summary_stats = users_filtered[all_metrics].describe()
    print(summary_stats.round(2))

print(f"\nOverview analysis complete. Results saved in: {OVERVIEW_DIR}")