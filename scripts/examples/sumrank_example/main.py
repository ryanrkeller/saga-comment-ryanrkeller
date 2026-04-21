import argparse
import sys
import os
sys.path.insert(0, '/Users/Pokem/Coleman Research/saga-comment-ryanrkeller/saga-comment-ryanrkeller/src')

import random
import csv
from typing import Dict, List, Tuple
import time
import pandas as pd
from itertools import product
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import pathlib

from saga.utils.random_graphs import get_branching_dag, get_network
from saga import Network, TaskGraph, Schedule
from saga.schedulers.heft import HeftScheduler, upward_rank, sum_rank, hybrid_rank
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph

thisdir = pathlib.Path(__file__).parent
output_dir = thisdir / "outputs"

def get_problem_instance() -> Tuple[Network, TaskGraph]:
    print("Creating network...")
    network = get_network()
    print("Creating task graph...")
    task_graph = get_branching_dag(
        levels=random.randint(2, 3),
        branching_factor=random.randint(2, 3)
    )
    print("Problem instance created!")
    return network, task_graph

def run_comparison(num_trials: int = 1000, use_sum_rank: bool = True) -> List[Dict]:
    """Run comparison between upward_rank and sum_rank."""
    results = []
    
    for i in range(num_trials):
        print(f"Trial {i+1}/{num_trials}: Generating problem instance...")
        network, task_graph = get_problem_instance()
        
        print(f"Trial {i+1}: Running upward_rank scheduler...")
        
        # Test upward_rank
        scheduler_upward = HeftScheduler()
        schedule_upward = scheduler_upward.schedule(network, task_graph)
        upward_makespan = schedule_upward.makespan
        print(f"Trial {i+1}: Upward makespan: {upward_makespan}")
        
        # Test sum_rank (if enabled)
        if use_sum_rank:
            print(f"Trial {i+1}: Running sum_rank scheduler...")
            # Temporarily replace upward_rank with sum_rank
            original_upward_rank = HeftScheduler.schedule.__globals__['upward_rank']
            HeftScheduler.schedule.__globals__['upward_rank'] = sum_rank
            
            print(f"Trial {i+1}: Creating scheduler...")
            scheduler_sum = HeftScheduler()
            schedule_sum = scheduler_sum.schedule(network, task_graph)
            sum_makespan = schedule_sum.makespan
            print(f"Trial {i+1}: Sum makespan: {sum_makespan}")
            
            # Restore original
            HeftScheduler.schedule.__globals__['upward_rank'] = original_upward_rank
        else: 
            sum_makespan = None
        
        result = {
            'trial': i,
            'num_tasks': len(task_graph.graph.nodes),
            'num_edges': len(task_graph.graph.edges),
            'upward_makespan': upward_makespan,
            'sum_makespan': sum_makespan,
            'improvement': (upward_makespan - sum_makespan) / upward_makespan * 100 if use_sum_rank and sum_makespan else None
        }
        
        results.append(result)
        
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{num_trials} trials")
    
    return results

def save_results(results: List[Dict], filename: str = "sumrank_comparison.csv"):
    """Save results to CSV file with better organization and readability."""
    if not results:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = output_dir / filename
    
    # Enhanced headers for better organization
    fieldnames = [
        'trial',
        'graph_size_tasks', 
        'graph_size_edges',
        'task_edge_ratio',
        'upward_rank_makespan',
        'sum_rank_makespan',
        'performance_difference',
        'performance_improvement_percent',
        'better_algorithm'
    ]
    
    # Transform results for better organization
    organized_results = []
    for r in results:
        if r['sum_makespan'] is not None:
            improvement = r['improvement']
            better = 'sum_rank' if improvement > 0 else 'upward_rank' if improvement < 0 else 'equal'
        else:
            improvement = None
            better = 'upward_rank'
            
        organized_results.append({
            'trial': r['trial'],
            'graph_size_tasks': r['num_tasks'],
            'graph_size_edges': r['num_edges'],
            'task_edge_ratio': round(r['num_tasks'] / r['num_edges'], 3) if r['num_edges'] > 0 else 0,
            'upward_rank_makespan': round(r['upward_makespan'], 6),
            'sum_rank_makespan': round(r['sum_makespan'], 6) if r['sum_makespan'] else None,
            'performance_difference': round(r['sum_makespan'] - r['upward_makespan'], 6) if r['sum_makespan'] else None,
            'performance_improvement_percent': round(improvement, 2) if improvement is not None else None,
            'better_algorithm': better
        })
    
    # Create a more readable format with labels
    readable_filepath = os.path.join("outputs", filename.replace('.csv', '_readable.csv'))
    
    with open(readable_filepath, 'w', newline='') as csvfile:
        csvfile.write("# HEFT Scheduler Comparison Results\n")
        csvfile.write("# Format: trial | graph_info | upward_rank | sum_rank | performance\n")
        csvfile.write("# " + "="*80 + "\n\n")
        
        for r in organized_results:
            csvfile.write(f"=== Trial {r['trial']} ===\n")
            csvfile.write(f"Graph Info: Tasks: {r['graph_size_tasks']:2d}, Edges: {r['graph_size_edges']:2d}, Ratio: {r['task_edge_ratio']:.2f}\n")
            csvfile.write(f"Upward Rank: {r['upward_rank_makespan']:8.6f}\n")
            if r['sum_rank_makespan']:
                csvfile.write(f"Sum Rank:    {r['sum_rank_makespan']:8.6f}\n")
                diff_str = f"{r['performance_difference']:+.6f}"
                csvfile.write(f"Difference:   {diff_str} ({r['performance_improvement_percent']:+.1f}%)\n")
                csvfile.write(f"Winner:       {r['better_algorithm']}\n")
            else:
                csvfile.write("Sum Rank:    N/A\n")
                csvfile.write("Difference:   N/A\n")
                csvfile.write("Winner:       upward_rank\n")
            csvfile.write("\n")
    
    # Also save the standard CSV for data analysis
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(organized_results)
    
    print(f"Results saved to {filepath}")
    print(f"Readable format saved to {readable_filepath}")
    print(f"Enhanced CSV with {len(organized_results)} trials and detailed metrics")

def analyze_results(results: List[Dict]):
    """Print analysis of results."""
    if not results:
        return
    
    upward_makespans = [r['upward_rank_makespan'] for r in results]
    sum_makespans = [r['sum_rank_makespan'] for r in results if r['sum_rank_makespan'] is not None]
    improvements = [r['performance_improvement_percent'] for r in results if r['performance_improvement_percent'] is not None]
    
    print(f"\n=== Analysis Results ===")
    print(f"Total trials: {len(results)}")
    print(f"Average upward makespan: {sum(upward_makespans)/len(upward_makespans):.2f}")
    
    if sum_makespans:
        print(f"Average sum makespan: {sum(sum_makespans)/len(sum_makespans):.2f}")
        print(f"Average improvement: {sum(improvements)/len(improvements):.2f}%")
        print(f"Trials where sum_rank better: {len([i for i in improvements if i > 0])}/{len(improvements)}")
        print(f"Max improvement: {max(improvements):.2f}%")
        print(f"Min improvement: {min(improvements):.2f}%")

        # Enhanced analysis: when does sum_rank perform better?
        better_trials = [r for r in results if r['performance_improvement_percent'] and r['performance_improvement_percent'] > 0]
        worse_trials = [r for r in results if r['performance_improvement_percent'] and r['performance_improvement_percent'] <= 0]
        
        if better_trials:
            print(f"\n=== When sum_rank BETTER ===")
            print(f"Number of better trials: {len(better_trials)}")
            avg_tasks_better = sum(r['graph_size_tasks'] for r in better_trials) / len(better_trials)
            avg_edges_better = sum(r['graph_size_edges'] for r in better_trials) / len(better_trials)
            print(f"Average tasks: {avg_tasks_better:.1f}")
            print(f"Average edges: {avg_edges_better:.1f}")
            print(f"Task/Edge ratio: {avg_tasks_better/avg_edges_better:.2f}")
        
        if worse_trials:
            print(f"\n=== When sum_rank WORSE ===")
            print(f"Number of worse trials: {len(worse_trials)}")
            avg_tasks_worse = sum(r['graph_size_tasks'] for r in worse_trials) / len(worse_trials)
            avg_edges_worse = sum(r['graph_size_edges'] for r in worse_trials) / len(worse_trials)
            print(f"Average tasks: {avg_tasks_worse:.1f}")
            print(f"Average edges: {avg_edges_worse:.1f}")
            print(f"Task/Edge ratio: {avg_tasks_worse/avg_edges_worse:.2f}")

def plot_results(results: List[Dict]):
    if not results:
        print("No results to plot!")
        return
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figure with subplots - 2x3 layout for more plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SumRank vs UpwardRank Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Performance Difference Distribution
    differences = [r['performance_improvement_percent'] for r in results if r['performance_improvement_percent'] is not None]
    axes[0, 0].hist(differences, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    axes[0, 0].set_xlabel('Performance Difference (%)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Performance Differences')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Task/Edge Ratio vs Performance
    task_edge_ratios = [r['task_edge_ratio'] for r in results if r['performance_improvement_percent'] is not None]
    performance_diffs = [r['performance_improvement_percent'] for r in results if r['performance_improvement_percent'] is not None]
    
    scatter = axes[0, 1].scatter(task_edge_ratios, performance_diffs, alpha=0.6, c=performance_diffs, cmap='RdYlGn', s=50)
    axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Task/Edge Ratio')
    axes[0, 1].set_ylabel('Performance Difference (%)')
    axes[0, 1].set_title('Task/Edge Ratio vs Performance')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[0, 1])
    cbar.set_label('Performance Difference (%)')
    
    # 3. Better vs Worse Trials
    better_count = len([r for r in results if r['performance_improvement_percent'] and r['performance_improvement_percent'] > 0])
    worse_count = len([r for r in results if r['performance_improvement_percent'] and r['performance_improvement_percent'] < 0])
    equal_count = len([r for r in results if r['performance_improvement_percent'] and r['performance_improvement_percent'] == 0])
    
    labels = ['Better (sum_rank)', 'Worse (sum_rank)', 'Equal']
    sizes = [better_count, worse_count, equal_count]
    colors = ['#2E8B57', '#CD5C5C', '#808080']
    
    axes[0, 2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0, 2].set_title('Performance Comparison Distribution')
    
    # 4. Average Performance by Task/Edge Ratio
    task_edge_bins = {}
    for r in results:
        ratio = round(r['task_edge_ratio'], 1)
        if ratio not in task_edge_bins:
            task_edge_bins[ratio] = {'sum': [], 'upward': []}
        task_edge_bins[ratio]['sum'].append(r['sum_rank_makespan'])
        task_edge_bins[ratio]['upward'].append(r['upward_rank_makespan'])
    
    ratios = sorted(task_edge_bins.keys())
    avg_sum = [np.mean(task_edge_bins[r]['sum']) for r in ratios]
    avg_upward = [np.mean(task_edge_bins[r]['upward']) for r in ratios]
    
    axes[1, 0].plot(ratios, avg_sum, 'o-', label='SumRank', linewidth=2, markersize=6)
    axes[1, 0].plot(ratios, avg_upward, 's-', label='UpwardRank', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Task/Edge Ratio')
    axes[1, 0].set_ylabel('Average Makespan')
    axes[1, 0].set_title('Average Makespan by Task/Edge Ratio')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. NEW: Makespan by Number of Tasks (Nodes) - Enhanced with better/worse indicators
    task_bins = {}
    for r in results:
        tasks = r['graph_size_tasks']
        if tasks not in task_bins:
            task_bins[tasks] = {'sum': [], 'upward': [], 'improvements': []}
        task_bins[tasks]['sum'].append(r['sum_rank_makespan'])
        task_bins[tasks]['upward'].append(r['upward_rank_makespan'])
        task_bins[tasks]['improvements'].append(r['performance_improvement_percent'])
    
    task_counts = sorted(task_bins.keys())
    avg_sum_tasks = [np.mean(task_bins[t]['sum']) for t in task_counts]
    avg_upward_tasks = [np.mean(task_bins[t]['upward']) for t in task_counts]
    
    # Calculate which algorithm is better for each task count
    better_algorithm = []
    colors = []
    for t in task_counts:
        avg_improvement = np.mean(task_bins[t]['improvements'])
        if avg_improvement > 1.0:  # sum_rank significantly better
            better_algorithm.append('sum_rank')
            colors.append('#2E8B57')  # green
        elif avg_improvement < -1.0:  # upward_rank significantly better
            better_algorithm.append('upward_rank')
            colors.append('#CD5C5C')  # red
        else:  # roughly equal
            better_algorithm.append('equal')
            colors.append('#808080')  # gray
    
    # Plot lines with background colors to show winner
    for i, (t, color) in enumerate(zip(task_counts, colors)):
        if i < len(task_counts) - 1:
            next_t = task_counts[i + 1]
            axes[1, 1].axvspan(t - 0.5, next_t - 0.5, alpha=0.2, color=color)
    
    axes[1, 1].plot(task_counts, avg_sum_tasks, 'o-', label='SumRank', linewidth=3, markersize=8, color='#2E8B57')
    axes[1, 1].plot(task_counts, avg_upward_tasks, 's-', label='UpwardRank', linewidth=3, markersize=8, color='#CD5C5C')
    axes[1, 1].set_xlabel('Number of Tasks (Nodes)')
    axes[1, 1].set_ylabel('Average Makespan')
    axes[1, 1].set_title('Makespan by Tasks (Green=SumRank Better, Red=UpwardRank Better)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add text annotations for significant differences
    for i, t in enumerate(task_counts):
        avg_improvement = np.mean(task_bins[t]['improvements'])
        if abs(avg_improvement) > 2.0:  # Only show significant differences
            winner = "SumRank" if avg_improvement > 0 else "UpwardRank"
            axes[1, 1].annotate(f'{winner}\n+{abs(avg_improvement):.1f}%', 
                               xy=(t, max(avg_sum_tasks[i], avg_upward_tasks[i])),
                               xytext=(t, max(avg_sum_tasks[i], avg_upward_tasks[i]) + 0.3),
                               ha='center', fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7))
    
    # 6. NEW: Makespan by Number of Edges (proxy for branching factor) - Enhanced
    edge_bins = {}
    for r in results:
        edges = r['graph_size_edges']
        if edges not in edge_bins:
            edge_bins[edges] = {'sum': [], 'upward': [], 'improvements': []}
        edge_bins[edges]['sum'].append(r['sum_rank_makespan'])
        edge_bins[edges]['upward'].append(r['upward_rank_makespan'])
        edge_bins[edges]['improvements'].append(r['performance_improvement_percent'])
    
    edge_counts = sorted(edge_bins.keys())
    avg_sum_edges = [np.mean(edge_bins[e]['sum']) for e in edge_counts]
    avg_upward_edges = [np.mean(edge_bins[e]['upward']) for e in edge_counts]
    
    # Calculate which algorithm is better for each edge count
    edge_colors = []
    for e in edge_counts:
        avg_improvement = np.mean(edge_bins[e]['improvements'])
        if avg_improvement > 1.0:  # sum_rank significantly better
            edge_colors.append('#2E8B57')  # green
        elif avg_improvement < -1.0:  # upward_rank significantly better
            edge_colors.append('#CD5C5C')  # red
        else:  # roughly equal
            edge_colors.append('#808080')  # gray
    
    # Plot lines with background colors
    for i, (e, color) in enumerate(zip(edge_counts, edge_colors)):
        if i < len(edge_counts) - 1:
            next_e = edge_counts[i + 1]
            axes[1, 2].axvspan(e - 0.5, next_e - 0.5, alpha=0.2, color=color)
    
    axes[1, 2].plot(edge_counts, avg_sum_edges, 'o-', label='SumRank', linewidth=3, markersize=8, color='#2E8B57')
    axes[1, 2].plot(edge_counts, avg_upward_edges, 's-', label='UpwardRank', linewidth=3, markersize=8, color='#CD5C5C')
    axes[1, 2].set_xlabel('Number of Edges (Branching Complexity)')
    axes[1, 2].set_ylabel('Average Makespan')
    axes[1, 2].set_title('Makespan by Edges (Green=SumRank Better, Red=UpwardRank Better)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add text annotations for significant differences
    for i, e in enumerate(edge_counts):
        avg_improvement = np.mean(edge_bins[e]['improvements'])
        if abs(avg_improvement) > 2.0:  # Only show significant differences
            winner = "SumRank" if avg_improvement > 0 else "UpwardRank"
            axes[1, 2].annotate(f'{winner}\n+{abs(avg_improvement):.1f}%', 
                               xy=(e, max(avg_sum_edges[i], avg_upward_edges[i])),
                               xytext=(e, max(avg_sum_edges[i], avg_upward_edges[i]) + 0.3),
                               ha='center', fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=edge_colors[i], alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sumrank_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved analysis plot to {output_dir / 'sumrank_analysis.png'}")
    plt.close()



def test_hybrid_alpha_values(num_trials: int = 30, alphas: List[float] = None):
    """Test different alpha values for hybrid ranking."""
    if alphas is None:
        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    results = []
    all_results = []  # Initialize to store detailed results
    
    print("Testing alpha values: ", end="")
    for alpha in alphas:
        print(f"{alpha:.1f} ", end="", flush=True)
        
        alpha_results = []
        
        for trial in range(num_trials):
            # Generate problem instance
            network, task_graph = get_problem_instance()
            
            # Test hybrid ranking with specified alpha
            original_upward_rank = HeftScheduler.schedule.__globals__['upward_rank']
            HeftScheduler.schedule.__globals__['upward_rank'] = lambda n, t: hybrid_rank(n, t, alpha)
            
            scheduler = HeftScheduler()
            schedule = scheduler.schedule(network, task_graph)
            hybrid_makespan = schedule.makespan
            
            # Restore original
            HeftScheduler.schedule.__globals__['upward_rank'] = original_upward_rank
            
            # Test baseline upward_rank for comparison
            scheduler_upward = HeftScheduler()
            schedule_upward = scheduler_upward.schedule(network, task_graph)
            upward_makespan = schedule_upward.makespan
            
            # Calculate improvement
            improvement = (upward_makespan - hybrid_makespan) / upward_makespan * 100
            
            alpha_results.append({
                'trial': trial,
                'alpha': alpha,
                'hybrid_makespan': hybrid_makespan,
                'upward_makespan': upward_makespan,
                'improvement_percent': improvement
            })
        
        # Calculate average performance for this alpha
        avg_improvement = sum(r['improvement_percent'] for r in alpha_results) / len(alpha_results)
        better_count = len([r for r in alpha_results if r['improvement_percent'] > 0])
        worse_count = len([r for r in alpha_results if r['improvement_percent'] < 0])
        
        results.append({
            'alpha': alpha,
            'avg_improvement_percent': avg_improvement,
            'better_trials': better_count,
            'worse_trials': worse_count,
            'equal_trials': num_trials - better_count - worse_count,
            'win_rate_percent': (better_count / num_trials) * 100
        })
        
        print(f"Alpha {alpha:.1f}: {avg_improvement:+.2f}% avg, {better_count}/{num_trials} better")
        
        # Store detailed results for plotting
        all_results.extend(alpha_results)
    
    return pd.DataFrame(results), all_results

def plot_alpha_analysis(df: pd.DataFrame, all_alpha_results: List[Dict]):
    """Plot the results of alpha value testing with enhanced readability."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Hybrid Rank Alpha Value Analysis - Finding the Optimal Sweet Spot', fontsize=18, fontweight='bold')
    
    # Set better default styles
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 16
    })
    
    # 1. Average Makespan Comparison by Alpha
    ax1 = axes[0, 0]
    
    # Calculate average makespans for each alpha
    avg_hybrid_makespan = []
    avg_upward_makespan = []
    
    for alpha in sorted(df['alpha'].unique()):
        alpha_data = [r for r in all_alpha_results if r['alpha'] == alpha]
        avg_hybrid = sum(r['hybrid_makespan'] for r in alpha_data) / len(alpha_data)
        avg_upward = sum(r['upward_makespan'] for r in alpha_data) / len(alpha_data)
        avg_hybrid_makespan.append(avg_hybrid)
        avg_upward_makespan.append(avg_upward)
    
    alphas_sorted = sorted(df['alpha'].unique())
    
    # Plot both makespan lines
    line1 = ax1.plot(alphas_sorted, avg_hybrid_makespan, 'o-', linewidth=3, markersize=10, 
                     color='#2E8B57', markerfacecolor='#2E8B57', markeredgecolor='darkgreen', 
                     markeredgewidth=2, label='Hybrid Makespan')
    line2 = ax1.plot(alphas_sorted, avg_upward_makespan, 's-', linewidth=3, markersize=10, 
                     color='#DC143C', markerfacecolor='#DC143C', markeredgecolor='darkred', 
                     markeredgewidth=2, label='UpwardRank Makespan')
    
    # Highlight optimal alpha (minimum makespan difference)
    makespan_diffs = [h - u for h, u in zip(avg_hybrid_makespan, avg_upward_makespan)]
    best_idx = makespan_diffs.index(min(makespan_diffs))
    best_alpha = alphas_sorted[best_idx]
    
    ax1.scatter([best_alpha], [avg_hybrid_makespan[best_idx]], color='gold', s=300, zorder=5, 
               edgecolor='darkgreen', linewidth=3, label=f'OPTIMAL: Alpha={best_alpha:.1f}')
    
    # Add makespan difference annotation
    diff_text = f'Best Alpha: {best_alpha:.1f}\nDiff: {makespan_diffs[best_idx]:+.3f}'
    ax1.annotate(diff_text,
                xy=(best_alpha, avg_hybrid_makespan[best_idx]), 
                xytext=(best_alpha + 0.1, avg_hybrid_makespan[best_idx] + 0.5),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
                fontsize=11, fontweight='bold')
    
    ax1.set_xlabel('Alpha Value\n(0.0 = Pure SumRank, 1.0 = Pure UpwardRank)', fontsize=12)
    ax1.set_ylabel('Average Makespan', fontsize=12)
    ax1.set_title('Makespan Comparison: Hybrid vs UpwardRank', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    
    # 2. Makespan Win Rate by Alpha
    ax2 = axes[0, 1]
    bars = ax2.bar(df['alpha'], df['win_rate_percent'], width=0.08, 
                   color=['#2E8B57' if wr >= 50 else '#CD5C5C' for wr in df['win_rate_percent']],
                   alpha=0.8, edgecolor='black', linewidth=1)
    ax2.axhline(50, color='blue', linestyle='--', linewidth=2, alpha=0.8, label='50% Baseline')
    ax2.axhline(df.loc[best_idx, 'win_rate_percent'], color='gold', linestyle='-', linewidth=3, 
                alpha=0.8, label=f'Best: {df.loc[best_idx, "win_rate_percent"]:.1f}%')
    
    # Add value labels on bars
    for i, (alpha, win_rate) in enumerate(zip(df['alpha'], df['win_rate_percent'])):
        if alpha == best_alpha:
            ax2.text(alpha, win_rate + 2, f'{win_rate:.1f}%', ha='center', va='bottom', 
                    fontweight='bold', color='darkgreen', fontsize=11)
        else:
            ax2.text(alpha, win_rate + 1, f'{win_rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Alpha Value\n(0.0 = Pure SumRank, 1.0 = Pure UpwardRank)', fontsize=12)
    ax2.set_ylabel('Win Rate vs UpwardRank (%)', fontsize=12)
    ax2.set_title('Win Rate by Alpha Value', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc='best', fontsize=10)
    ax2.set_ylim(0, max(df['win_rate_percent']) + 10)
    
    # 3. Trial Distribution by Alpha - Side-by-Side (Worse, Equal, Better)
    ax3 = axes[1, 0]
    width = 0.025  # Narrower width for 3 bars
    x = df['alpha']
    
    # Side-by-side in order: Worse (left), Equal (middle), Better (right)
    bars1 = ax3.bar(x - width, df['worse_trials'], width, label='Worse', color='#CD5C5C', alpha=0.8, edgecolor='black')
    bars2 = ax3.bar(x, df['equal_trials'], width, label='Equal', color='#808080', alpha=0.8, edgecolor='black')
    bars3 = ax3.bar(x + width, df['better_trials'], width, label='Better', color='#2E8B57', alpha=0.8, edgecolor='black')
    
    # Add value labels on top of each bar
    for i, alpha in enumerate(df['alpha']):
        # Worse (left)
        if df['worse_trials'].iloc[i] > 0:
            ax3.text(alpha - width, df['worse_trials'].iloc[i] + 0.5, 
                    f'{df["worse_trials"].iloc[i]:.0f}', ha='center', va='bottom', fontweight='bold', color='#CD5C5C', fontsize=9)
        
        # Equal (middle)
        if df['equal_trials'].iloc[i] > 0:
            ax3.text(alpha, df['equal_trials'].iloc[i] + 0.5, 
                    f'{df["equal_trials"].iloc[i]:.0f}', ha='center', va='bottom', fontweight='bold', color='#808080', fontsize=9)
        
        # Better (right)
        if df['better_trials'].iloc[i] > 0:
            ax3.text(alpha + width, df['better_trials'].iloc[i] + 0.5, 
                    f'{df["better_trials"].iloc[i]:.0f}', ha='center', va='bottom', fontweight='bold', color='#2E8B57', fontsize=9)
    
    # Highlight optimal alpha
    best_idx = df['avg_improvement_percent'].idxmax()
    best_alpha = df.loc[best_idx, 'alpha']
    ax3.axvline(best_alpha, color='gold', linestyle='--', linewidth=3, alpha=0.8, label=f'Optimal Alpha={best_alpha:.1f}')
    
    ax3.set_xlabel('Alpha Value\n(0.0 = Pure SumRank, 1.0 = Pure UpwardRank)', fontsize=12)
    ax3.set_ylabel('Number of Trials (out of 30)', fontsize=12)
    ax3.set_title('Trial Outcome Distribution by Alpha', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(loc='best', fontsize=10)
    ax3.set_xticks(df['alpha'])
    
    # 4. Performance Summary - Enhanced
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create enhanced summary text with better formatting
    best_alpha = df.loc[df['avg_improvement_percent'].idxmax(), 'alpha']
    best_improvement = df.loc[df['avg_improvement_percent'].idxmax(), 'avg_improvement_percent']
    best_win_rate = df.loc[df['avg_improvement_percent'].idxmax(), 'win_rate_percent']
    best_better = df.loc[df['avg_improvement_percent'].idxmax(), 'better_trials']
    best_equal = df.loc[df['avg_improvement_percent'].idxmax(), 'equal_trials']
    best_worse = df.loc[df['avg_improvement_percent'].idxmax(), 'worse_trials']
    
    # Determine recommendation
    if best_alpha < 0.3:
        recommendation = "Use SumRank-Heavy Approach"
        rec_color = '#2E8B57'
    elif best_alpha > 0.7:
        recommendation = "Use UpwardRank-Heavy Approach"
        rec_color = '#CD5C5C'
    else:
        recommendation = "Use Balanced Approach"
        rec_color = '#4169E1'
    
    summary_text = f"""
    {'='*50}
    OPTIMAL ALPHA FOUND: {best_alpha:.1f}
    {'='*50}
    
    PERFORMANCE METRICS:
    Average Improvement: {best_improvement:+.2f}%
    Win Rate: {best_win_rate:.1f}%
    
    TRIAL BREAKDOWN (30 trials):
    Better:   {best_better:.0f} trials ({best_better/30*100:.1f}%)
    Equal:    {best_equal:.0f} trials ({best_equal/30*100:.1f}%)
    Worse:    {best_worse:.0f} trials ({best_worse/30*100:.1f}%)
    
    ALPHA INTERPRETATION:
    0.0 = Pure SumRank (total work focus)
    0.5 = Equal Balance
    1.0 = Pure UpwardRank (critical path focus)
    
    {'='*50}
    RECOMMENDATION:
    {recommendation}
    {'='*50}
    """
    
    ax4.text(0.05, 0.95, summary_text, fontsize=11, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.02', facecolor='lightblue', alpha=0.3))
    
    # Add recommendation box
    rec_text = f"RECOMMENDED:\n{recommendation}\nAlpha = {best_alpha:.1f}"
    ax4.text(0.75, 0.5, rec_text, fontsize=14, fontweight='bold',
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round,pad=0.02', facecolor=rec_color, alpha=0.8, edgecolor='black', linewidth=2),
            color='white')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hybrid_alpha_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved enhanced alpha analysis plot to {output_dir / 'hybrid_alpha_analysis.png'}")
    plt.close()

def test_alpha_simple():
    """
    Simple alpha testing function - change the alpha value inside this function.
    
    Just modify the alpha variable below and run this function to test different values.
    """
    # ===== CHANGE THIS ALPHA VALUE =====
    alpha = 0.2  # Modify this value (0.0 = pure sum_rank, 1.0 = pure upward_rank)
    # ====================================
    
    num_trials = 10
    print(f"\n=== Testing Alpha {alpha:.1f} ===")
    print(f"Running {num_trials} trials...")
    
    results = []
    
    for trial in range(num_trials):
        # Generate problem instance
        network, task_graph = get_problem_instance()
        
        # Test hybrid ranking with specified alpha
        original_upward_rank = HeftScheduler.schedule.__globals__['upward_rank']
        HeftScheduler.schedule.__globals__['upward_rank'] = lambda n, t: hybrid_rank(n, t, alpha)
        
        scheduler = HeftScheduler()
        schedule = scheduler.schedule(network, task_graph)
        hybrid_makespan = schedule.makespan
        
        # Restore original
        HeftScheduler.schedule.__globals__['upward_rank'] = original_upward_rank
        
        # Test baseline upward_rank for comparison
        scheduler_upward = HeftScheduler()
        schedule_upward = scheduler_upward.schedule(network, task_graph)
        upward_makespan = schedule_upward.makespan
        
        # Calculate improvement
        improvement = (upward_makespan - hybrid_makespan) / upward_makespan * 100
        
        result = {
            'trial': trial + 1,
            'alpha': alpha,
            'hybrid_makespan': hybrid_makespan,
            'upward_makespan': upward_makespan,
            'improvement_percent': improvement,
            'num_tasks': len(task_graph.graph.nodes),
            'num_edges': len(task_graph.graph.edges),
            'task_edge_ratio': len(task_graph.graph.nodes) / len(task_graph.graph.edges) if len(task_graph.graph.edges) > 0 else 0
        }
        
        results.append(result)
        
        status = "BETTER" if improvement > 0 else "WORSE" if improvement < 0 else "EQUAL"
        print(f"Trial {trial+1:2d}: {status} | Hybrid: {hybrid_makespan:7.3f} | Upward: {upward_makespan:7.3f} | Diff: {improvement:+6.2f}%")
    
    # Calculate summary
    improvements = [r['improvement_percent'] for r in results]
    better_count = len([r for r in results if r['improvement_percent'] > 0])
    worse_count = len([r for r in results if r['improvement_percent'] < 0])
    equal_count = len([r for r in results if r['improvement_percent'] == 0])
    
    avg_improvement = sum(improvements) / len(improvements)
    
    print(f"\n--- Alpha {alpha:.1f} Summary ---")
    print(f"Average Improvement: {avg_improvement:+.2f}%")
    print(f"Win Rate: {better_count/num_trials*100:.1f}% ({better_count}/{num_trials})")
    print(f"Trial Breakdown: {better_count} better, {equal_count} equal, {worse_count} worse")
    
    # Performance assessment
    if avg_improvement > 1.0:
        assessment = "EXCELLENT - Significant improvement"
    elif avg_improvement > 0.5:
        assessment = "GOOD - Noticeable improvement"
    elif avg_improvement > 0:
        assessment = "SLIGHT - Minor improvement"
    elif avg_improvement > -0.5:
        assessment = "NEUTRAL - Similar performance"
    else:
        assessment = "POOR - Performance degradation"
    
    print(f"Assessment: {assessment}")
    
    # Save results to CSV
    csv_filename = output_dir / f'alpha_{alpha:.1f}_results.csv'
    readable_filename = output_dir / f'alpha_{alpha:.1f}_results_readable.csv'
    
    # Save detailed CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_filename, index=False)
    print(f"Detailed results saved to {csv_filename}")
    
    # Create readable format
    with open(readable_filename, 'w') as f:
        f.write(f"Alpha {alpha:.1f} Test Results\n")
        f.write(f"{'='*50}\n")
        f.write(f"Average Improvement: {avg_improvement:+.2f}%\n")
        f.write(f"Win Rate: {better_count/num_trials*100:.1f}% ({better_count}/{num_trials})\n")
        f.write(f"Assessment: {assessment}\n")
        f.write(f"\nDetailed Results:\n")
        f.write(f"{'='*50}\n")
        
        for result in results:
            status = "BETTER" if result['improvement_percent'] > 0 else "WORSE" if result['improvement_percent'] < 0 else "EQUAL"
            f.write(f"\nTrial {result['trial']}:\n")
            f.write(f"  Status: {status}\n")
            f.write(f"  Graph: {result['num_tasks']} tasks, {result['num_edges']} edges (ratio: {result['task_edge_ratio']:.2f})\n")
            f.write(f"  Hybrid Makespan: {result['hybrid_makespan']:.3f}\n")
            f.write(f"  Upward Makespan: {result['upward_makespan']:.3f}\n")
            f.write(f"  Improvement: {result['improvement_percent']:+.2f}%\n")
        
        f.write(f"\n{'='*50}\n")
        f.write(f"Summary Statistics:\n")
        f.write(f"Total Trials: {num_trials}\n")
        f.write(f"Better: {better_count} ({better_count/num_trials*100:.1f}%)\n")
        f.write(f"Equal: {equal_count} ({equal_count/num_trials*100:.1f}%)\n")
        f.write(f"Worse: {worse_count} ({worse_count/num_trials*100:.1f}%)\n")
        f.write(f"Average Improvement: {avg_improvement:+.2f}%\n")
        f.write(f"Max Improvement: {max(improvements):+.2f}%\n")
        f.write(f"Min Improvement: {min(improvements):+.2f}%\n")
    
    print(f"Readable results saved to {readable_filename}")
    
    return {
        'alpha': alpha,
        'avg_improvement': avg_improvement,
        'win_rate': better_count/num_trials*100,
        'better_trials': better_count,
        'worse_trials': worse_count,
        'equal_trials': equal_count
    }

def analyze_hybrid_sweet_spot():
    """Run complete analysis to find optimal hybrid alpha."""
    print("=== Finding Hybrid Rank Sweet Spot ===")
    
    # Test alpha values
    results_df, all_results = test_hybrid_alpha_values(num_trials=30, alphas=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    # Save results
    results_df.to_csv(output_dir / 'hybrid_alpha_results.csv', index=False)
    print(f"Alpha results saved to {output_dir / 'hybrid_alpha_results.csv'}")
    
    # Plot analysis
    plot_alpha_analysis(results_df, all_results)
    
    # Print summary
    best_row = results_df.loc[results_df['avg_improvement_percent'].idxmax()]
    print(f"\n=== OPTIMAL ALPHA FOUND ===")
    print(f"Best Alpha: {best_row['alpha']:.1f}")
    print(f"Average Improvement: {best_row['avg_improvement_percent']:+.2f}%")
    print(f"Win Rate: {best_row['win_rate_percent']:.1f}%")
    print(f"Better/Equal/Worse: {best_row['better_trials']}/{best_row['equal_trials']}/{best_row['worse_trials']}")
    
    return results_df

def visualize_schedules(results: List[Dict], num_examples: int = 5):
    
    if not better_examples and not worse_examples:
        print("No significant performance differences found for visualization")
        return
    
    for i, result in enumerate(better_examples):
        print(f"Creating charts for better example {i+1} (improvement: {result['improvement']:.1f}%)")
        
        # Generate the same problem instance
        network, task_graph = get_problem_instance()
        
        # Create schedules
        scheduler_upward = HeftScheduler()
        schedule_upward = scheduler_upward.schedule(network, task_graph)
        
        # Temporarily replace upward_rank with sum_rank
        original_upward_rank = HeftScheduler.schedule.__globals__['upward_rank']
        HeftScheduler.schedule.__globals__['upward_rank'] = sum_rank
        scheduler_sum = HeftScheduler()
        schedule_sum = scheduler_sum.schedule(network, task_graph)
        HeftScheduler.schedule.__globals__['upward_rank'] = original_upward_rank
        
        # Draw Gantt charts only with trial number
        trial_num = result['trial']
        ax = draw_gantt(schedule_upward.mapping, use_latex=False)
        fig = ax.get_figure()
        if fig:
            fig.savefig(str(savedir / f'trial_{trial_num}_better_example_{i+1}_upward_chart.png'))
            plt.close(fig)
        
        ax = draw_gantt(schedule_sum.mapping, use_latex=False)
        fig = ax.get_figure()
        if fig:
            fig.savefig(str(savedir / f'trial_{trial_num}_better_example_{i+1}_sum_chart.png'))
            plt.close(fig)
    
    # Visualize examples where sum_rank performs worse
    for i, result in enumerate(worse_examples):
        print(f"Creating charts for worse example {i+1} (improvement: {result['improvement']:.1f}%)")
        
        # Generate the same problem instance
        network, task_graph = get_problem_instance()
        
        # Create schedules
        scheduler_upward = HeftScheduler()
        schedule_upward = scheduler_upward.schedule(network, task_graph)
        
        # Temporarily replace upward_rank with sum_rank
        original_upward_rank = HeftScheduler.schedule.__globals__['upward_rank']
        HeftScheduler.schedule.__globals__['upward_rank'] = sum_rank
        scheduler_sum = HeftScheduler()
        schedule_sum = scheduler_sum.schedule(network, task_graph)
        HeftScheduler.schedule.__globals__['upward_rank'] = original_upward_rank
        
        # Draw Gantt charts only with trial number
        trial_num = result['trial']
        ax = draw_gantt(schedule_upward.mapping, use_latex=False)
        fig = ax.get_figure()
        if fig:
            fig.savefig(str(savedir / f'trial_{trial_num}_worse_example_{i+1}_upward_chart.png'))
            plt.close(fig)
        
        ax = draw_gantt(schedule_sum.mapping, use_latex=False)
        fig = ax.get_figure()
        if fig:
            fig.savefig(str(savedir / f'trial_{trial_num}_worse_example_{i+1}_sum_chart.png'))
            plt.close(fig)
    
    print(f"- Charts saved to {savedir} directory")
    print(f"- Compare upward vs sum_rank scheduling decisions")
    print(f"- See task ordering and processor assignment differences")

def run_experiment():
    num_instances = 20
    ccr_values = [1/10, 5, 10]
    duplicate_factors = [1, 2, 3]
    all_num_nodes = [4, 8]
    all_levels = [2, 3]
    all_branching_factors = [2, 3]

    schedulers = {
        "HEFT_Original": lambda dup: HeftScheduler(),
        "HEFT_SumRank": lambda dup: HeftScheduler()  # We'll modify this to use sum_rank
    }

    # Calculate total iterations for progress bar
    total_iterations = (
        num_instances *
        len(ccr_values) *
        len(list(product(all_num_nodes, all_levels, all_branching_factors))) *
        len(duplicate_factors) *
        len(schedulers)
    )

    rows = []
    with tqdm(total=total_iterations, desc="Running experiments") as pbar:
        for i in range(num_instances):
            for ccr in ccr_values:
                for num_nodes, levels, branching_factor in product(all_num_nodes, all_levels, all_branching_factors):
                    network, task_graph = get_branching_dag(
                        levels=levels,
                        branching_factor=branching_factor
                    )
                    
                    for dup_factor in duplicate_factors:
                        for scheduler_name, scheduler_func in schedulers.items():
                            if "SumRank" in scheduler_name:
                                # Temporarily replace upward_rank with sum_rank
                                original_upward_rank = HeftScheduler.schedule.__globals__['upward_rank']
                                HeftScheduler.schedule.__globals__['upward_rank'] = sum_rank
                                scheduler = scheduler_func(dup_factor)
                                test_schedule = scheduler.schedule(network, task_graph)
                                # Restore original
                                HeftScheduler.schedule.__globals__['upward_rank'] = original_upward_rank
                            else:
                                scheduler = scheduler_func(dup_factor)
                                test_schedule = scheduler.schedule(network, task_graph)
                            
                            makespan = test_schedule.makespan
                            rows.append([i, ccr, num_nodes, levels, branching_factor, scheduler_name, dup_factor, makespan])
                            pbar.update(1)

    df = pd.DataFrame(rows, columns=["Instance", "CCR", "Num Nodes", "Levels", "Branching Factor", "Scheduler", "Dup Factor", "Makespan"])
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/experiment_results.csv", index=False)
    print(f"Results saved to outputs/experiment_results.csv")
    return df


def main():
    print("Starting sum_rank vs upward_rank comparison...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, default="run", choices=["run", "analyze", "visualize_schedules", "hybrid_analysis", "simple_alpha"])
    args = parser.parse_args()
    
    results = None

    # Run comparison
    if args.command == "run":
        results = run_comparison(num_trials=50, use_sum_rank=True)
        
        # Save results
        save_results(results)
    
    # Analyze results
    if args.command == "analyze":
        results = pd.read_csv(output_dir / "sumrank_comparison.csv")
        results_dict = results.to_dict('records')
        analyze_results(results_dict)
        plot_results(results_dict)
    
    # Create chart visualizations
    if args.command == "visualize_schedules":
        results = pd.read_csv(output_dir / "sumrank_comparison.csv")
        results_dict = results.to_dict('records')
        print("\nCreating chart visualizations...")
        visualize_schedules(results_dict, num_examples=2)
    
    # Hybrid analysis to find optimal alpha
    if args.command == "hybrid_analysis":
        analyze_hybrid_sweet_spot()
    
    # Simple alpha testing
    if args.command == "simple_alpha":
        test_alpha_simple()
    
    print("\nDone!")

if __name__ == "__main__":
    main()