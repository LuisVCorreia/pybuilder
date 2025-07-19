from typing import List
from queue import PriorityQueue
import logging

from .task import ConflictGroup, TaskPriority, get_tasks_for_group

logger = logging.getLogger(__name__)


class ConflictTaskGenerator:
    """Simplified task generator for backtesting - no change tracking or existing groups."""
    
    def __init__(self, task_queue: PriorityQueue):
        self.task_queue = task_queue

    def generate_tasks_for_groups(self, conflict_groups: List[ConflictGroup]):
        """Generate and queue tasks for all conflict groups."""
        total_tasks = 0
        
        for group in conflict_groups:
            tasks = get_tasks_for_group(group, TaskPriority.HIGH)
            
            for task in tasks:
                self.task_queue.put(task)
                total_tasks += 1
        
        logger.info(f"Generated {total_tasks} tasks for {len(conflict_groups)} conflict groups")
        return total_tasks
