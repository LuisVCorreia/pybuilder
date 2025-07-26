from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
from functools import total_ordering
import time
from .groups import ConflictGroup

@total_ordering
class TaskPriority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def display(self) -> str:
        return {
            TaskPriority.LOW: "Low",
            TaskPriority.MEDIUM: "Medium", 
            TaskPriority.HIGH: "High"
        }[self]


class Algorithm(Enum):
    GREEDY = "greedy"
    REVERSE_GREEDY = "reverse_greedy"
    LENGTH = "length"
    ALL_PERMUTATIONS = "all_permutations"
    RANDOM = "random"


@dataclass
class RandomConfig:
    seed: int
    count: int


@dataclass
class ResolutionResult:
    """Result of resolving a conflict group - ordered sequence with total profit."""
    total_profit: int  # Total coinbase profit in wei
    sequence_of_orders: List[Tuple[int, int]]  # List of (order_index, profit) tuples


@dataclass 
@total_ordering
class ConflictTask:
    """Task for resolving a ConflictGroup with a specific Algorithm."""
    group_idx: int
    algorithm: Algorithm
    priority: TaskPriority
    group: ConflictGroup
    created_at: float
    random_config: RandomConfig = None

    def __lt__(self, other):
        if not isinstance(other, ConflictTask):
            return NotImplemented
        # Higher priority first, then earlier created_at for ties
        if self.priority != other.priority:
            return self.priority > other.priority  # Reverse for higher priority first
        return self.created_at < other.created_at

    def __eq__(self, other):
        if not isinstance(other, ConflictTask):
            return NotImplemented
        return (self.priority == other.priority and 
                abs(self.created_at - other.created_at) < 1e-9)

# Constants matching rbuilder
MAX_LENGTH_FOR_ALL_PERMUTATIONS = 3
NUMBER_OF_RANDOM_TASKS = 10  # Note: rbuilder uses 50


def get_tasks_for_group(group: ConflictGroup, priority: TaskPriority) -> List[ConflictTask]:
    tasks = []
    created_at = time.time()
    
    # Always run greedy first for quick decent results
    tasks.append(ConflictTask(
        group_idx=group.id,
        algorithm=Algorithm.GREEDY,
        priority=priority,
        group=group,
        created_at=created_at
    ))
    
    # Add all permutations if small enough
    if len(group.orders) <= MAX_LENGTH_FOR_ALL_PERMUTATIONS:
        tasks.append(ConflictTask(
            group_idx=group.id,
            algorithm=Algorithm.ALL_PERMUTATIONS,
            priority=priority,
            group=group,
            created_at=created_at
        ))
    else:
        # Add reverse greedy for larger groups
        tasks.append(ConflictTask(
            group_idx=group.id,
            algorithm=Algorithm.REVERSE_GREEDY,
            priority=TaskPriority.LOW,
            group=group,
            created_at=created_at
        ))
        
        # Add length-based algorithm
        tasks.append(ConflictTask(
            group_idx=group.id,
            algorithm=Algorithm.LENGTH,
            priority=TaskPriority.LOW,
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
            random_config=RandomConfig(seed=group.id, count=NUMBER_OF_RANDOM_TASKS)
        ))
    
    return tasks
