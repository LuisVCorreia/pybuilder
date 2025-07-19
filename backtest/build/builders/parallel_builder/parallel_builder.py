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
from .task import ConflictTask

logger = logging.getLogger(__name__)


def _task_execution_wrapper(
    context,
    rpc_url: str,
    task: ConflictTask,
) -> SimulatedOrder:
    """
    Per-task worker executed in a thread.

    Creates a fresh EVMSimulator (which forks at the required parent block).
    Caching + prev-hash handling are now managed by patched titanoboa (boa_ext),
    so we do NOT create per-thread cache dirs or manually copy prev hashes.
    """
    # Each thread creates its own simulator. The underlying disk cache file
    # can be shared safely (independent sqlite connections per thread).
    simulator = EVMSimulator(
        simulation_context=context,
        rpc_url=rpc_url
    )

    resolver = ConflictResolver(simulator)
    return resolver.resolve_conflict_task(task)


class ParallelBuilderConfig:
    """Configuration for the parallel builder."""
    def __init__(self, num_threads: int = 4):
        self.num_threads = num_threads


class ParallelBuilder:
    """
    Parallel block builder:

    1. Group orders by state-trace conflicts
    2. Generate tasks for each conflict group
    3. Resolve tasks concurrently
    4. Aggregate best results
    5. Assemble final block
    """

    def __init__(self, config: ParallelBuilderConfig | None = None):
        self.config = config or ParallelBuilderConfig()
        self.builder_name = "parallel"

    def build_block(
        self,
        simulated_orders: List[SimulatedOrder],
        evm_simulator: EVMSimulator
    ) -> BlockResult:
        start_time = time.time()
        try:
            logger.info(
                "Starting parallel block building with %d orders",
                len(simulated_orders)
            )

            simulated_orders.sort(key=lambda o: o.order.id().value)

            # Step 1: Conflict grouping
            conflict_finder = ConflictFinder()
            conflict_groups = conflict_finder.add_orders(simulated_orders)
            logger.info("Found %d conflict groups", len(conflict_groups))
            self._log_group_stats(conflict_groups)

            # Step 2: Create tasks
            task_queue = PriorityQueue()
            task_generator = ConflictTaskGenerator(task_queue)
            num_tasks = task_generator.generate_tasks_for_groups(conflict_groups)
            logger.info("Generated %d tasks", num_tasks)

            # Step 3: Aggregators
            best_results = BestResults()
            results_aggregator = ResultsAggregator(best_results)

            # Step 4: Parallel resolution
            self._resolve_conflicts_parallel(
                task_queue,
                evm_simulator,
                results_aggregator
            )

            # Step 5: Assemble block
            current_results = results_aggregator.get_current_best_results()
            block_assembler = BlockAssembler(self.builder_name)
            block_result = block_assembler.assemble_block(current_results)

            elapsed_ms = (time.time() - start_time) * 1000
            if block_result.block_trace:
                block_result.block_trace.fill_time_ms = elapsed_ms
            block_result.build_time_ms = elapsed_ms

            stats = results_aggregator.get_stats()
            logger.info(
                "Parallel builder done in %.2f ms | results processed=%d | final profit=%.6f ETH",
                elapsed_ms,
                stats["results_processed"],
                stats["total_profit"] / 1e18
            )

            return block_result

        except Exception as e:
            logger.error("Error in parallel builder: %s", e, exc_info=True)
            return BlockResult(
                builder_name=self.builder_name,
                success=False,
                error_message=str(e),
                build_time_ms=(time.time() - start_time) * 1000
            )

    def _resolve_conflicts_parallel(
        self,
        task_queue: PriorityQueue,
        evm_simulator: EVMSimulator,
        results_aggregator: ResultsAggregator
    ):
        """Drain the task queue and process tasks concurrently."""
        tasks: List[ConflictTask] = []
        while not task_queue.empty():
            try:
                tasks.append(task_queue.get_nowait())
            except Empty:
                break

        total = len(tasks)
        logger.info(
            "Processing %d tasks with %d threads",
            total,
            self.config.num_threads
        )

        # Immutable data reused by workers
        context = evm_simulator.context
        rpc_url = evm_simulator.rpc_url

        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            future_to_task = {
                executor.submit(_task_execution_wrapper, context, rpc_url, task): task
                for task in tasks
            }

            for idx, future in enumerate(as_completed(future_to_task), 1):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results_aggregator.update_result(
                        task.group_idx, result, task.group
                    )
                except Exception as e:
                    logger.warning(
                        "Task failed for group %s: %s",
                        task.group_idx,
                        e,
                        exc_info=True
                    )

                if idx % 10 == 0 or idx == total:
                    logger.debug("Completed %d/%d tasks", idx, total)

    def _log_group_stats(self, conflict_groups):
        if not conflict_groups:
            return
        single_order_groups = sum(1 for g in conflict_groups if len(g.orders) == 1)
        multi_order_groups = len(conflict_groups) - single_order_groups
        sizes = [len(g.orders) for g in conflict_groups]
        max_size = max(sizes, default=0)
        avg_size = sum(sizes) / len(sizes)
        logger.info(
            "Group stats: %d single-order, %d multi-order | max=%d avg=%.1f",
            single_order_groups,
            multi_order_groups,
            max_size,
            avg_size
        )
        logger.debug("Group sizes: %s", sizes)


def run_parallel_builder(
    simulated_orders: List[SimulatedOrder],
    config: Dict,
    evm_simulator: EVMSimulator
) -> BlockResult:
    """
    Entrypoint to execute the parallel builder based on `config`.
    """
    builder_configs = {b['name']: b for b in config.get('builders', [])}
    parallel_cfg = builder_configs.get('parallel', {})
    num_threads = parallel_cfg.get('num_threads', 4)

    builder = ParallelBuilder(ParallelBuilderConfig(num_threads=num_threads))
    return builder.build_block(simulated_orders, evm_simulator)
