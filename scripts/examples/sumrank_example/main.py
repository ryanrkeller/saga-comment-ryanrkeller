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

from saga.utils.random_graphs import get_branching_dag, get_network
from saga import Network, TaskGraph, Schedule
from saga.schedulers.heft import HeftScheduler, upward_rank, sum_rank
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph

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
    
    os.makedirs("outputs", exist_ok=True)
    filepath = os.path.join("outputs", filename)
    
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
    
    upward_makespans = [r['upward_makespan'] for r in results]
    sum_makespans = [r['sum_makespan'] for r in results if r['sum_makespan'] is not None]
    improvements = [r['improvement'] for r in results if r['improvement'] is not None]
    
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
        better_trials = [r for r in results if r['improvement'] and r['improvement'] > 0]
        worse_trials = [r for r in results if r['improvement'] and r['improvement'] <= 0]
        
        if better_trials:
            print(f"\n=== When sum_rank BETTER ===")
            print(f"Number of better trials: {len(better_trials)}")
            avg_tasks_better = sum(r['num_tasks'] for r in better_trials) / len(better_trials)
            avg_edges_better = sum(r['num_edges'] for r in better_trials) / len(better_trials)
            print(f"Average tasks: {avg_tasks_better:.1f}")
            print(f"Average edges: {avg_edges_better:.1f}")
            print(f"Task/Edge ratio: {avg_tasks_better/avg_edges_better:.2f}")
        
        if worse_trials:
            print(f"\n=== When sum_rank WORSE ===")
            print(f"Number of worse trials: {len(worse_trials)}")
            avg_tasks_worse = sum(r['num_tasks'] for r in worse_trials) / len(worse_trials)
            avg_edges_worse = sum(r['num_edges'] for r in worse_trials) / len(worse_trials)
            print(f"Average tasks: {avg_tasks_worse:.1f}")
            print(f"Average edges: {avg_edges_worse:.1f}")
            print(f"Task/Edge ratio: {avg_tasks_worse/avg_edges_worse:.2f}")

def visualize_schedules(results: List[Dict], num_examples: int = 3):
    """Create Gantt chart visualizations of sample schedules using saga's built-in functions."""
    if not results:
        print("No results to visualize!")
        return
    
    print(f"\n=== Creating Gantt Chart Visualizations ===")
    # Use the same outputs directory as basic_example
    savedir = Path(__file__).parent / 'outputs'
    savedir.mkdir(exist_ok=True)
    
    # Find interesting examples
    better_examples = [r for r in results if r['improvement'] and r['improvement'] > 0][:num_examples]
    worse_examples = [r for r in results if r['improvement'] and r['improvement'] < 0][:num_examples]
    
    if not better_examples and not worse_examples:
        print("No significant performance differences found for visualization")
        return
    
    # Visualize examples where sum_rank performs better
    for i, result in enumerate(better_examples):
        print(f"Creating Gantt charts for better example {i+1} (improvement: {result['improvement']:.1f}%)")
        
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
        
        # Draw Gantt charts only
        ax = draw_gantt(schedule_upward.mapping, use_latex=False)
        fig = ax.get_figure()
        if fig:
            fig.savefig(str(savedir / f'better_example_{i+1}_upward_gantt.png'))
            plt.close(fig)
        
        ax = draw_gantt(schedule_sum.mapping, use_latex=False)
        fig = ax.get_figure()
        if fig:
            fig.savefig(str(savedir / f'better_example_{i+1}_sum_gantt.png'))
            plt.close(fig)
    
    # Visualize examples where sum_rank performs worse
    for i, result in enumerate(worse_examples):
        print(f"Creating Gantt charts for worse example {i+1} (improvement: {result['improvement']:.1f}%)")
        
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
        
        # Draw Gantt charts only
        ax = draw_gantt(schedule_upward.mapping, use_latex=False)
        fig = ax.get_figure()
        if fig:
            fig.savefig(str(savedir / f'worse_example_{i+1}_upward_gantt.png'))
            plt.close(fig)
        
        ax = draw_gantt(schedule_sum.mapping, use_latex=False)
        fig = ax.get_figure()
        if fig:
            fig.savefig(str(savedir / f'worse_example_{i+1}_sum_gantt.png'))
            plt.close(fig)
    
    print(f"Gantt charts saved to {savedir} directory")
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
    
    # Run comparison
    results = run_comparison(num_trials=25, use_sum_rank=True)
    
    # Save results
    save_results(results)
    
    # Analyze results
    analyze_results(results)
    
    # Create Gantt chart visualizations
    print("\nCreating Gantt chart visualizations...")
    visualize_schedules(results, num_examples=2)
    
    print("\nDone! Check outputs/ directory for all results and visualizations.")
    print("- CSV files: sumrank_comparison.csv, sumrank_comparison_readable.csv")
    print("- Gantt charts: Compare upward vs sum_rank scheduling decisions")

if __name__ == "__main__":
    main()