import threading
from typing import Dict, List, Tuple, Optional
import logging

from .task import ConflictGroup, ResolutionResult

logger = logging.getLogger(__name__)


class BestResults:
    """Thread-safe storage of best results for each conflict group."""
    
    def __init__(self):
        self.data: Dict[int, Tuple[ResolutionResult, ConflictGroup]] = {}
        self.version = 0
        self._lock = threading.RLock()

    def update_result(self, group_id: int, result: ResolutionResult, group: ConflictGroup) -> bool:
        """Update best result for a group. Returns True if updated."""
        with self._lock:
            existing = self.data.get(group_id)
            
            if existing is None or result.total_profit > existing[0].total_profit:
                old_profit = existing[0].total_profit if existing else 0
                self.data[group_id] = (result, group)
                self.version += 1
                
                logger.debug(
                    f"Updated best result for group {group_id}: "
                    f"{old_profit} -> {result.total_profit} wei "
                    f"(+{result.total_profit - old_profit} wei)"
                )
                return True
            
            return False

    def get_results_and_version(self) -> Tuple[List[Tuple[ResolutionResult, ConflictGroup]], int]:
        """Get all current best results and version."""
        with self._lock:
            return list(self.data.values()), self.version

    def get_number_of_orders(self) -> int:
        """Get total number of orders across all groups."""
        with self._lock:
            return sum(len(group.orders) for _, group in self.data.values())

    def get_version(self) -> int:
        """Get current version."""
        with self._lock:
            return self.version

    def get_total_profit(self) -> int:
        """Get sum of all best results."""
        with self._lock:
            return sum(result.total_profit for result, _ in self.data.values())


class ResultsAggregator:
    """Collects and manages best results from conflict resolution tasks."""
    
    def __init__(self, best_results: BestResults):
        self.best_results = best_results
        self.results_processed = 0

    def update_result(self, group_id: int, result: ResolutionResult, group: ConflictGroup) -> bool:
        """Update best result for a group."""
        self.results_processed += 1
        
        updated = self.best_results.update_result(group_id, result, group)
        
        if updated:
            logger.info(
                f"New best result for group {group_id} with {len(group.orders)} orders: "
                f"{result.total_profit / 1e18:.6f} ETH profit. "
                f"Total results processed: {self.results_processed}"
            )
        
        return updated

    def get_current_best_results(self) -> List[Tuple[ResolutionResult, ConflictGroup]]:
        """Get current best results for all groups."""
        results, _ = self.best_results.get_results_and_version()
        return results

    def get_stats(self) -> Dict[str, int]:
        """Get aggregator statistics."""
        return {
            'results_processed': self.results_processed,
            'total_groups': len(self.best_results.data),
            'total_orders': self.best_results.get_number_of_orders(),
            'total_profit': self.best_results.get_total_profit(),
            'version': self.best_results.get_version()
        }
