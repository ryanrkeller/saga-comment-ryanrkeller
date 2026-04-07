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
from saga.schedulers.heft import HeftScheduler, upward_rank, sum_rank
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
    
    # 5. NEW: Makespan by Number of Tasks (Nodes)
    task_bins = {}
    for r in results:
        tasks = r['graph_size_tasks']
        if tasks not in task_bins:
            task_bins[tasks] = {'sum': [], 'upward': []}
        task_bins[tasks]['sum'].append(r['sum_rank_makespan'])
        task_bins[tasks]['upward'].append(r['upward_rank_makespan'])
    
    task_counts = sorted(task_bins.keys())
    avg_sum_tasks = [np.mean(task_bins[t]['sum']) for t in task_counts]
    avg_upward_tasks = [np.mean(task_bins[t]['upward']) for t in task_counts]
    
    axes[1, 1].plot(task_counts, avg_sum_tasks, 'o-', label='SumRank', linewidth=2, markersize=6)
    axes[1, 1].plot(task_counts, avg_upward_tasks, 's-', label='UpwardRank', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Number of Tasks (Nodes)')
    axes[1, 1].set_ylabel('Average Makespan')
    axes[1, 1].set_title('Average Makespan by Number of Tasks')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. NEW: Makespan by Number of Edges (proxy for branching factor)
    edge_bins = {}
    for r in results:
        edges = r['graph_size_edges']
        if edges not in edge_bins:
            edge_bins[edges] = {'sum': [], 'upward': []}
        edge_bins[edges]['sum'].append(r['sum_rank_makespan'])
        edge_bins[edges]['upward'].append(r['upward_rank_makespan'])
    
    edge_counts = sorted(edge_bins.keys())
    avg_sum_edges = [np.mean(edge_bins[e]['sum']) for e in edge_counts]
    avg_upward_edges = [np.mean(edge_bins[e]['upward']) for e in edge_counts]
    
    axes[1, 2].plot(edge_counts, avg_sum_edges, 'o-', label='SumRank', linewidth=2, markersize=6)
    axes[1, 2].plot(edge_counts, avg_upward_edges, 's-', label='UpwardRank', linewidth=2, markersize=6)
    axes[1, 2].set_xlabel('Number of Edges (Branching Complexity)')
    axes[1, 2].set_ylabel('Average Makespan')
    axes[1, 2].set_title('Average Makespan by Number of Edges')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sumrank_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved analysis plot to {output_dir / 'sumrank_analysis.png'}")
    plt.close()



def visualize_schedules(results: List[Dict], num_examples: int = 5):
    if not results:
        print("No results to visualize!")
        return
    
    print(f"\n=== Creating Chart Visualizations ===")
    savedir = Path(__file__).parent / 'outputs'
    savedir.mkdir(exist_ok=True)
    
    better_examples = [r for r in results if r['improvement'] and r['improvement'] > 0][:num_examples]
    worse_examples = [r for r in results if r['improvement'] and r['improvement'] < 0][:num_examples]
    
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
    parser.add_argument("--command", type=str, default="run", choices=["run", "analyze", "visualize_schedules"])
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
    
    print("\nDone!")

if __name__ == "__main__":
    main()