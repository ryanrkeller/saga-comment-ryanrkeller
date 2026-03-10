import pathlib
from typing import List, Optional
import numpy as np

from saga import Schedule, Scheduler, ScheduledTask, TaskGraph, Network
from saga.schedulers.cpop import upward_rank

thisdir = pathlib.Path(__file__).resolve().parent

def calculate_task_difficulty(task_graph: TaskGraph, task_name: str) -> float:
    difficulty = 0.0

    parents = list(task_graph.graph.predecessors(task_name))
    while parents:
        parent = parents.pop(0)
        difficulty += task_graph.get_task(parent).cost
        parents.extend(task_graph.graph.predecessors(parent))
    
    return difficulty

def heft_rank_sort(network: Network, task_graph: TaskGraph) -> List[str]:
    """Sort tasks based on their rank (as defined in the HEFT paper).
    Args:
        network (Network): The network graph.
        task_graph (TaskGraph): The task graph.

    Returns:
        List[str]: The sorted list of tasks.
    """
    urank = upward_rank(network, task_graph) # Upward rank calculation (HEFT part 1)
    # urank = average comp time + max(child + comp time)
        # represents longest path from current task to finish point
    
    topological_sort = {
        node.name: i for i, node in enumerate(reversed(task_graph.topological_sort()))
    } 
    # makes a dictionary of teach task and its urank, used as 
    # tiebreaker if 2 tasks have the same urank value,
    # task closer to the finish point is done first

    # Old tiebreak: tiebreak by which one is closer to the finish point
    # rank = {node: (urank[node], topological_sort[node]) for node in urank} 

    
    # # rank of a task is how long it takes to get from the starting node to the finishing node 
    # # (includes the time to complete the task itself)
    # Calculate difficulty for each task
    task_difficulties = {node: calculate_task_difficulty(task_graph, node) for node in urank}

    # new ranking: (upward_rank, difficulty, topological_position)
    rank = {
        node: (
            urank[node],                           # first: upward rank
            task_difficulties[node],               # second: task difficulty  
            topological_sort[node]                 # third: position in chain
        ) 
        for node in urank
    }


    # ---------- sort nodes in a list by order of urank values ----------
    order = sorted(list(rank.keys()), key=lambda x: rank.get(x, 0.0), reverse=True)
    # sort by urank value (descending by order of urank values, uses randomness to break ties)
    
    return order


class HeftScheduler(Scheduler):
    """Schedules tasks using the HEFT algorithm.

    Source: https://dx.doi.org/10.1109/71.993206
    """

    def schedule(
        self,
        network: Network,
        task_graph: TaskGraph,
        schedule: Optional[Schedule] = None,
        min_start_time: float = 0.0,
    ) -> Schedule:
        """Schedule the tasks on the network.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            schedule (Optional[Dict[Hashable, List[Task]]], optional): The schedule. Defaults to None.
            min_start_time (float, optional): The minimum start time. Defaults to 0.0.

        Returns:
            Dict[str, List[Task]]: The schedule.

        Raises:
            ValueError: If the instance is invalid.
        """
        schedule_order = heft_rank_sort(network, task_graph)
        # sort by decending order of urank values, tiebreak by which task is closer to the finish point
        schedule = Schedule(task_graph, network)
        # creates an empty schedule that gets filled as tasks get assigned to processors

        for task_name in schedule_order:
            if schedule.is_scheduled(task_name):
                continue
            min_finish_time = np.inf # initialize min_finish_time to infinity
            # in the end it keeps track of the minimum finish time
            best_node = next(iter(network.nodes))  # initialize best_node to a random node
            # in the end it keeps track of the processor that results in the minimum finish time

# ---------- assign tasks to processors with least computational time respectively ----------
            
            for node in network.nodes:

                # EST calculation
                start_time = schedule.get_earliest_start_time(
                    task=task_name, node=node, append_only=False
                #calculates when task can start based on 
                # - dependencies
                # - communication times
                # - processor availability
                )
                start_time = max(start_time, min_start_time)
                # makes sure that the task doesn't start before min_start_time
                
                # Runtime calculation: task cost divided by network speed
                runtime = (
                    task_graph.get_task(task_name).cost / network.get_node(node).speed
                )
                
                # Finish time calculation: start time + runtime
                finish_time = start_time + runtime
                if finish_time < min_finish_time:
                    min_finish_time = finish_time
                    # if finish time is faster set mintime to this node's runtime
                    
                    # best_start_time = start_time
                
                    best_node = node
                    # set best node to this node
            new_task = ScheduledTask(
            # creates final scheduletask object with optimal task assignment
                node=best_node.name,
                # processor that gives earliest finish time
                name=task_name,
                # current task being scheduled
                start=min_finish_time
                - (
                    task_graph.get_task(task_name).cost
                    / network.get_node(best_node).speed
                ),
                # start time is finish time minus runtime
                # this could be saved above in the finish time calculation
                end=min_finish_time,
                # end time is finish time
            )
            schedule.add_task(new_task)
            # adds task to schedule

        return schedule
        # returns final schedule
