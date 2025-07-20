from typing import List
from queue import PriorityQueue
import logging
from backtest.build.builders.parallel_builder.results_aggregator import ResultsAggregator
from .task import ConflictGroup, TaskPriority, get_tasks_for_group, ResolutionResult

logger = logging.getLogger(__name__)


class ConflictTaskGenerator:
    """Simplified task generator for backtesting - no change tracking or existing groups."""
    
    def __init__(self, task_queue: PriorityQueue, results_aggregator: ResultsAggregator):
        self.task_queue = task_queue
        self.results_aggregator = results_aggregator

    def generate_tasks_for_groups(self, conflict_groups: List[ConflictGroup]):
        """Generate and queue tasks for all conflict groups."""
        total_tasks = 0
        
        for group in conflict_groups:
            if len(group.orders) == 1:
                # No need to create tasks for single order groups, send to results aggregator
                total_profit = group.orders[0].sim_value.coinbase_profit
                result = ResolutionResult(
                    total_profit=total_profit,
                    sequence_of_orders=[(0, total_profit)]
                )
                self.results_aggregator.update_result(
                    group.id, result, group
                )
                continue

            tasks = get_tasks_for_group(group, TaskPriority.HIGH)
            
            for task in tasks:
                self.task_queue.put(task)
                total_tasks += 1
        
        logger.info(f"Generated {total_tasks} tasks for {len(conflict_groups)} conflict groups")
        return total_tasks
