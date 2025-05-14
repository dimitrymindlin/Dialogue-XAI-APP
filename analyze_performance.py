#!/usr/bin/env python3
"""
Performance comparison analysis script for LlamaIndex vs OpenAI Agents SDK.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import argparse

def parse_log_line(line):
    """Parse a single log line into a dictionary."""
    try:
        parts = line.strip().split(' - ', 1)
        if len(parts) < 2:
            return None
            
        timestamp, data = parts
        
        if 'PERFORMANCE_DATA' in data or 'TOTAL_TIME' in data:
            data_parts = data.split(',', 4)
            if len(data_parts) < 5:
                return None
                
            log_type, agent_type, time_str, exp_id, query = data_parts
            time_seconds = float(time_str.replace('s', ''))
            
            return {
                'timestamp': timestamp,
                'log_type': log_type,
                'agent_type': agent_type, 
                'time_seconds': time_seconds,
                'experiment_id': exp_id,
                'query': query.strip()
            }
        return None
    except Exception as e:
        print(f"Error parsing line: {line}, error: {e}")
        return None

def analyze_performance_logs(log_file='performance-logs/performance_comparison.log', output_dir='performance-reports'):
    """
    Analyze performance logs and generate reports.
    
    Args:
        log_file (str): Path to the log file
        output_dir (str): Directory to save reports
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Current timestamp for report filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Read log file
    if not os.path.exists(log_file):
        print(f"Error: Log file {log_file} does not exist.")
        return
    
    print(f"Reading log file: {log_file}")
    
    # Parse log file
    log_data = []
    with open(log_file, 'r') as f:
        for line in f:
            parsed_data = parse_log_line(line)
            if parsed_data:
                log_data.append(parsed_data)
    
    if not log_data:
        print("No valid data found in log file.")
        return
        
    # Create DataFrame
    df = pd.DataFrame(log_data)
    
    # Split into two dataframes: PERFORMANCE_DATA and TOTAL_TIME
    df_performance = df[df['log_type'] == 'PERFORMANCE_DATA']
    df_total = df[df['log_type'] == 'TOTAL_TIME']
    
    # Create summary reports
    create_summary_report(df_performance, df_total, output_dir, timestamp)
    
    # Create visualizations
    create_visualizations(df_performance, df_total, output_dir, timestamp)
    
    print(f"Analysis complete. Reports saved to {output_dir}")

def create_summary_report(df_performance, df_total, output_dir, timestamp):
    """Create summary reports from the dataframes."""
    # Generate summaries
    perf_stats = df_performance.groupby('agent_type')['time_seconds'].agg(['count', 'mean', 'median', 'min', 'max', 'std'])
    total_stats = df_total.groupby('agent_type')['time_seconds'].agg(['count', 'mean', 'median', 'min', 'max', 'std'])
    
    # Save to CSV
    perf_stats.to_csv(f"{output_dir}/performance_summary_{timestamp}.csv")
    total_stats.to_csv(f"{output_dir}/total_time_summary_{timestamp}.csv")
    
    # Create a human-readable report
    with open(f"{output_dir}/performance_report_{timestamp}.txt", 'w') as f:
        f.write("=== LlamaIndex vs OpenAI Agents SDK Performance Comparison ===\n\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("== PERFORMANCE_DATA Summary ==\n")
        f.write("This represents the time taken by each agent type for processing.\n\n")
        f.write(str(perf_stats))
        f.write("\n\n")
        
        # Calculate improvement percentage if both agent types exist
        if 'llama_index' in perf_stats.index and 'openai_agents' in perf_stats.index:
            llama_mean = perf_stats.loc['llama_index', 'mean']
            openai_mean = perf_stats.loc['openai_agents', 'mean']
            improvement = ((llama_mean - openai_mean) / llama_mean) * 100
            f.write(f"Improvement: OpenAI Agents is {improvement:.2f}% {'faster' if improvement > 0 else 'slower'} than LlamaIndex on average.\n\n")
        
        f.write("== TOTAL_TIME Summary ==\n")
        f.write("This represents the total time including all overhead.\n\n")
        f.write(str(total_stats))
        f.write("\n\n")
        
        # Calculate improvement percentage if both agent types exist
        if 'llama_index' in total_stats.index and 'openai_agents' in total_stats.index:
            llama_mean = total_stats.loc['llama_index', 'mean']
            openai_mean = total_stats.loc['openai_agents', 'mean']
            improvement = ((llama_mean - openai_mean) / llama_mean) * 100
            f.write(f"Improvement: OpenAI Agents is {improvement:.2f}% {'faster' if improvement > 0 else 'slower'} than LlamaIndex on average.\n\n")
    
    print(f"Summary report saved to {output_dir}/performance_report_{timestamp}.txt")

def create_visualizations(df_performance, df_total, output_dir, timestamp):
    """Create visualizations from the dataframes."""
    # Set the style
    sns.set(style="whitegrid")
    
    # Create performance comparison boxplot
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='agent_type', y='time_seconds', data=df_performance)
    plt.title('LlamaIndex vs OpenAI Agents SDK Processing Times')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Agent Type')
    
    # Add mean values as annotations
    means = df_performance.groupby('agent_type')['time_seconds'].mean()
    for i, agent_type in enumerate(means.index):
        plt.text(i, means[agent_type] * 1.05, f'Mean: {means[agent_type]:.2f}s', 
                horizontalalignment='center', size='small', color='black', weight='semibold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_boxplot_{timestamp}.png")
    
    # Create total time comparison boxplot
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='agent_type', y='time_seconds', data=df_total)
    plt.title('LlamaIndex vs OpenAI Agents SDK Total Times')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Agent Type')
    
    # Add mean values as annotations
    means = df_total.groupby('agent_type')['time_seconds'].mean()
    for i, agent_type in enumerate(means.index):
        plt.text(i, means[agent_type] * 1.05, f'Mean: {means[agent_type]:.2f}s', 
                horizontalalignment='center', size='small', color='black', weight='semibold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/total_time_boxplot_{timestamp}.png")
    
    # Create a bar chart comparison
    plt.figure(figsize=(12, 6))
    
    # Get the mean values
    perf_means = df_performance.groupby('agent_type')['time_seconds'].mean()
    total_means = df_total.groupby('agent_type')['time_seconds'].mean()
    
    # Set up bar positions
    bar_width = 0.35
    agent_types = list(set(perf_means.index) | set(total_means.index))
    indices = np.arange(len(agent_types))
    
    # Plot the bars
    perf_bars = plt.bar(indices - bar_width/2, [perf_means.get(agent, 0) for agent in agent_types], 
                        bar_width, label='Processing Time')
    total_bars = plt.bar(indices + bar_width/2, [total_means.get(agent, 0) for agent in agent_types], 
                        bar_width, label='Total Time')
    
    plt.xlabel('Agent Type')
    plt.ylabel('Time (seconds)')
    plt.title('LlamaIndex vs OpenAI Agents SDK: Processing vs Total Time')
    plt.xticks(indices, agent_types)
    plt.legend()
    
    # Add values on top of bars
    for i, bar in enumerate(perf_bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{bar.get_height():.2f}s', horizontalalignment='center', size='small')
    
    for i, bar in enumerate(total_bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{bar.get_height():.2f}s', horizontalalignment='center', size='small')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_comparison_bar_{timestamp}.png")
    
    # Create a time series plot if we have enough data
    if len(df_performance) > 5:
        plt.figure(figsize=(14, 7))
        
        # Convert timestamp to datetime
        df_performance['timestamp'] = pd.to_datetime(df_performance['timestamp'])
        df_performance = df_performance.sort_values('timestamp')
        
        # Plot time series
        sns.lineplot(x='timestamp', y='time_seconds', hue='agent_type', data=df_performance, markers=True)
        plt.title('Processing Time Over Time')
        plt.ylabel('Time (seconds)')
        plt.xlabel('Timestamp')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/time_series_{timestamp}.png")
    
    print(f"Visualizations saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Analyze performance logs comparing LlamaIndex and OpenAI Agents SDK.')
    parser.add_argument('--log_file', default='performance-logs/performance_comparison.log', 
                        help='Path to the performance log file')
    parser.add_argument('--output_dir', default='performance-reports', 
                        help='Directory to save analysis reports and visualizations')
    
    args = parser.parse_args()
    
    analyze_performance_logs(args.log_file, args.output_dir)

if __name__ == "__main__":
    main() 