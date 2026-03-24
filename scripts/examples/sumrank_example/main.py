import sys
import os
sys.path.insert(0, '/Users/Pokem/Coleman Research/saga-comment-ryanrkeller/saga-comment-ryanrkeller/src')

import random
import csv
from typing import Dict, List, Tuple
import time

from saga.utils.random_graphs import get_branching_dag, get_network
from saga import Network, TaskGraph, Schedule
from saga.schedulers.heft import HeftScheduler, upward_rank, sum_rank

def get_problem_instance() -> Tuple[Network, TaskGraph]:
    print("Creating network...")
    network = get_network()
    print("Creating task graph...")
    task_graph = get_branching_dag(
        levels=random.randint(5, 8),
        branching_factor=random.randint(3, 5)
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
            print(f"Trial {i+1}: Scheduling with sum_rank...")
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
    """Save results to CSV file."""
    if not results:
        return
    
    os.makedirs("outputs", exist_ok=True)
    filepath = os.path.join("outputs", filename)
    
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to {filepath}")

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

def main():
    print("Starting sum_rank vs upward_rank comparison...")
    
    # Run comparison
    results = run_comparison(num_trials=50, use_sum_rank=True)
    
    # Save results
    save_results(results)
    
    # Analyze results
    analyze_results(results)
    
    print("\nDone! Check outputs/sumrank_comparison.csv for detailed results.")

if __name__ == "__main__":
    main()