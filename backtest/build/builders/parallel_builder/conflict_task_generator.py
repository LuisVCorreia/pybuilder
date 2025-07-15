import time
from typing import Dict, List, Set
from queue import PriorityQueue
import logging

from .task import ConflictTask, ConflictGroup, TaskPriority, Algorithm, RandomConfig

logger = logging.getLogger(__name__)

# Constants matching rbuilder
THRESHOLD_FOR_SIGNIFICANT_CHANGE = 20
NUMBER_OF_TOP_ORDERS_TO_CONSIDER_FOR_SIGNIFICANT_CHANGE = 10
MAX_LENGTH_FOR_ALL_PERMUTATIONS = 3
NUMBER_OF_RANDOM_TASKS = 50


class ConflictTaskGenerator:
    """Manages conflicts and generates tasks for conflict groups."""
    
    def __init__(self, task_queue: PriorityQueue):
        self.existing_groups: Dict[int, ConflictGroup] = {}
        self.task_queue = task_queue

    def process_groups(self, new_groups: List[ConflictGroup]):
        """Process new conflict groups, updating existing groups or creating new tasks."""
        processed_groups = set()
        
        for new_group in new_groups:
            if self._has_been_processed(new_group.id, processed_groups):
                continue
                
            # Add this group and all conflicting groups to processed set
            self._add_processed_groups(processed_groups, new_group)
            
            # Remove any conflicting subset groups
            self._remove_conflicting_subset_groups(new_group)
            
            # Process the group
            self._process_single_group(new_group)

    def _add_processed_groups(self, processed_groups: Set[int], group: ConflictGroup):
        """Add group and all conflicting groups to processed set."""
        processed_groups.add(group.id)
        processed_groups.update(group.conflicting_group_ids)

    def _has_been_processed(self, group_id: int, processed_groups: Set[int]) -> bool:
        """Check if group has been processed."""
        return group_id in processed_groups

    def _remove_conflicting_subset_groups(self, superset_group: ConflictGroup):
        """Remove conflicting subset groups from existing groups."""
        to_remove = []
        for group_id in superset_group.conflicting_group_ids:
            if group_id in self.existing_groups:
                to_remove.append(group_id)
        
        for group_id in to_remove:
            del self.existing_groups[group_id]

    def _process_single_group(self, new_group: ConflictGroup):
        """Process a single group, determining priority and creating tasks."""
        if len(new_group.orders) == 1:
            # Single order groups can be processed immediately
            self._process_single_order_group(new_group.id, new_group)
        else:
            # Multi-order groups need task generation
            self._process_multi_order_group(new_group.id, new_group)

    def _process_single_order_group(self, group_id: int, group: ConflictGroup):
        """Process single order group - just return the order as best result."""
        # Single orders don't need complex resolution, but we still create a simple task
        task = ConflictTask(
            group_idx=group_id,
            algorithm=Algorithm.GREEDY,
            priority=TaskPriority.HIGH,
            group=group,
            created_at=time.time()
        )
        self.task_queue.put(task)
        self.existing_groups[group_id] = group

    def _process_multi_order_group(self, group_id: int, new_group: ConflictGroup):
        """Process multi-order group, determining priority and creating tasks."""
        existing_group = self.existing_groups.get(group_id)
        
        if existing_group is None:
            # New group - create tasks with medium priority
            priority = TaskPriority.MEDIUM
        else:
            # Check if this is a significant change
            priority = self._determine_update_priority(existing_group, new_group)
        
        # Generate and queue tasks for this group
        tasks = self._get_tasks_for_group(new_group, priority)
        for task in tasks:
            self.task_queue.put(task)
        
        # Update existing groups
        self.existing_groups[group_id] = new_group

    def _determine_update_priority(self, old_group: ConflictGroup, new_group: ConflictGroup) -> TaskPriority:
        """Determine priority based on significance of changes."""
        if len(old_group.orders) != len(new_group.orders):
            # Group size changed - this is significant
            return TaskPriority.HIGH
        
        # Check if top orders changed significantly
        old_top_profits = self._get_top_order_profits(old_group, NUMBER_OF_TOP_ORDERS_TO_CONSIDER_FOR_SIGNIFICANT_CHANGE)
        new_top_profits = self._get_top_order_profits(new_group, NUMBER_OF_TOP_ORDERS_TO_CONSIDER_FOR_SIGNIFICANT_CHANGE)
        
        for old_profit, new_profit in zip(old_top_profits, new_top_profits):
            if abs(old_profit - new_profit) > THRESHOLD_FOR_SIGNIFICANT_CHANGE:
                return TaskPriority.HIGH
        
        return TaskPriority.LOW

    def _get_top_order_profits(self, group: ConflictGroup, count: int) -> List[int]:
        """Get profits of top N orders by profit."""
        profits = [order.sim_value.coinbase_profit for order in group.orders]
        profits.sort(reverse=True)
        return profits[:count]

    def _get_tasks_for_group(self, group: ConflictGroup, priority: TaskPriority) -> List[ConflictTask]:
        """Generate tasks for a conflict group."""
        tasks = []
        created_at = time.time()
        
        # Always create greedy task (highest priority for actual priority)
        if priority == TaskPriority.HIGH:
            task_priority = TaskPriority.HIGH
        else:
            task_priority = TaskPriority.MEDIUM
            
        tasks.append(ConflictTask(
            group_idx=group.id,
            algorithm=Algorithm.GREEDY,
            priority=task_priority,
            group=group,
            created_at=created_at
        ))
        
        # Add reverse greedy task
        tasks.append(ConflictTask(
            group_idx=group.id,
            algorithm=Algorithm.REVERSE_GREEDY,
            priority=TaskPriority.LOW,
            group=group,
            created_at=created_at
        ))
        
        # Add length-based task
        tasks.append(ConflictTask(
            group_idx=group.id,
            algorithm=Algorithm.LENGTH,
            priority=TaskPriority.LOW,
            group=group,
            created_at=created_at
        ))
        
        # Add all permutations if small enough
        if len(group.orders) <= MAX_LENGTH_FOR_ALL_PERMUTATIONS:
            tasks.append(ConflictTask(
                group_idx=group.id,
                algorithm=Algorithm.ALL_PERMUTATIONS,
                priority=TaskPriority.MEDIUM,
                group=group,
                created_at=created_at
            ))
        
        # Add random permutation tasks
        tasks.append(ConflictTask(
            group_idx=group.id,
            algorithm=Algorithm.RANDOM,
            priority=TaskPriority.LOW,
            group=group,
            created_at=created_at,
            random_config=RandomConfig(seed=42, count=NUMBER_OF_RANDOM_TASKS)
        ))
        
        return tasks
