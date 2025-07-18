import time
import logging
from queue import PriorityQueue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

from backtest.build.simulation.sim_utils import SimulatedOrder
from backtest.build.simulation.evm_simulator import EVMSimulator
from backtest.build.builders.block_result import BlockResult

from .groups import ConflictFinder
from .conflict_task_generator import ConflictTaskGenerator
from .conflict_resolvers import ConflictResolver
from .results_aggregator import BestResults, ResultsAggregator
from .block_assembler import BlockAssembler

logger = logging.getLogger(__name__)


class ParallelBuilderConfig:
    """Configuration for the parallel builder."""
    
    def __init__(self, num_threads: int = 4):
        self.num_threads = num_threads


class ParallelBuilder:
    """
    Parallel block builder that groups conflicting orders and resolves them optimally.
    
    This mirrors rbuilder's parallel builder architecture:
    1. Groups orders by state trace conflicts
    2. Generates tasks for each conflict group
    3. Resolves conflicts in parallel using different algorithms
    4. Aggregates best results and assembles final block
    """
    
    def __init__(self, config: ParallelBuilderConfig = None):
        self.config = config or ParallelBuilderConfig()
        self.builder_name = "parallel"

    def build_block(self, simulated_orders: List[SimulatedOrder], evm_simulator: EVMSimulator) -> BlockResult:
        """Build a block using the parallel builder algorithm."""
        start_time = time.time()
        
        try:
            logger.info(f"Starting parallel block building with {len(simulated_orders)} orders")

            simulated_orders.sort(key=lambda o: o.order.id().value)
            
            # Step 1: Group orders by conflicts
            conflict_finder = ConflictFinder()
            conflict_groups = conflict_finder.add_orders(simulated_orders)
            
            logger.info(f"Found {len(conflict_groups)} conflict groups")
            self._log_group_stats(conflict_groups)
            
            # Step 2: Generate tasks for each group
            task_queue = PriorityQueue()
            task_generator = ConflictTaskGenerator(task_queue)
            num_tasks = task_generator.generate_tasks_for_groups(conflict_groups)

            logger.info(f"Generated {num_tasks} tasks")

            # Step 3: Set up result aggregation
            best_results = BestResults()
            results_aggregator = ResultsAggregator(best_results)
            
            # Step 4: Resolve conflicts in parallel
            self._resolve_conflicts_parallel(task_queue, evm_simulator, results_aggregator)
            
            # Step 5: Assemble final block
            current_results = results_aggregator.get_current_best_results()
            block_assembler = BlockAssembler(self.builder_name)
            block_result = block_assembler.assemble_block(current_results)
            
            # Set timing information
            build_time_ms = (time.time() - start_time) * 1000
            if block_result.block_trace:
                block_result.block_trace.fill_time_ms = build_time_ms
            block_result.build_time_ms = build_time_ms
            
            # Log final stats
            stats = results_aggregator.get_stats()
            logger.info(
                f"Parallel builder completed in {build_time_ms:.2f}ms. "
                f"Processed {stats['results_processed']} results, "
                f"final profit: {stats['total_profit'] / 1e18:.6f} ETH"
            )
            
            return block_result
            
        except Exception as e:
            logger.error(f"Error in parallel builder: {e}")
            return BlockResult(
                builder_name=self.builder_name,
                success=False,
                error_message=str(e),
                build_time_ms=(time.time() - start_time) * 1000
            )

    def _resolve_conflicts_parallel(self, task_queue: PriorityQueue, evm_simulator: EVMSimulator, 
                                  results_aggregator: ResultsAggregator):
        """Resolve conflicts using parallel task execution."""
        total_tasks = task_queue.qsize()
        
        # Collect all tasks first
        tasks = []
        while not task_queue.empty():
            try:
                tasks.append(task_queue.get_nowait())
            except Empty:
                break
        
        logger.info(f"Processing {len(tasks)} tasks with {self.config.num_threads} threads")
        
        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            # Submit all tasks
            future_to_task = {}
            for task in tasks:
                conflict_resolver = ConflictResolver(evm_simulator)
                future = executor.submit(conflict_resolver.resolve_conflict_task, task)
                future_to_task[future] = task
            
            # Process results as they complete
            completed_tasks = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                completed_tasks += 1
                
                try:
                    result = future.result()
                    results_aggregator.update_result(task.group_idx, result, task.group)
                    
                    if completed_tasks % 10 == 0:
                        logger.debug(f"Completed {completed_tasks}/{total_tasks} tasks")
                        
                except Exception as e:
                    logger.warning(f"Task failed for group {task.group_idx}: {e}")

    def _log_group_stats(self, conflict_groups):
        """Log statistics about conflict groups."""
        if not conflict_groups:
            return
            
        single_order_groups = sum(1 for g in conflict_groups if len(g.orders) == 1)
        multi_order_groups = len(conflict_groups) - single_order_groups
        max_group_size = max(len(g.orders) for g in conflict_groups)
        avg_group_size = sum(len(g.orders) for g in conflict_groups) / len(conflict_groups)
        
        logger.info(
            f"Group stats: {single_order_groups} single-order, {multi_order_groups} multi-order. "
            f"Max size: {max_group_size}, avg size: {avg_group_size:.1f}"
        )
        # log size of each group
        group_sizes = [len(g.orders) for g in conflict_groups]
        logger.info(f"Group sizes: {group_sizes}")


def run_parallel_builder(simulated_orders: List[SimulatedOrder], config: Dict, 
                        evm_simulator: EVMSimulator) -> BlockResult:
    """Run the parallel builder with given configuration."""
    # Get builder configurations
    builder_configs = {b['name']: b for b in config.get('builders', [])}
    parallel_config = builder_configs.get('parallel', {})
    
    # Extract config parameters
    num_threads = parallel_config.get('num_threads', 4)
    
    builder_config = ParallelBuilderConfig(num_threads=num_threads)
    builder = ParallelBuilder(builder_config)
    
    return builder.build_block(simulated_orders, evm_simulator)
