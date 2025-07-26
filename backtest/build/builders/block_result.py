import logging
from dataclasses import dataclass
from typing import List, Optional
from backtest.build.simulation.sim_utils import SimulatedOrder

logger = logging.getLogger(__name__)


@dataclass
class BlockTrace:
    """
    Block building trace information.
    """
    bid_value: int  # True bid value after subtracting payout gas cost (wei)
    gas_used: int
    gas_limit: int
    blob_gas_used: int = 0
    num_orders: int = 0
    orders_closed_at: float = 0.0
    fill_time_ms: float = 0.0
    raw_coinbase_profit: int = 0
    payout_gas_cost: int = 0


@dataclass  
class BlockResult:
    """
    Result of block building process.
    Contains the built block and associated metadata.
    """
    builder_name: str
    success: bool
    block_trace: Optional[BlockTrace] = None
    included_orders: Optional[List[SimulatedOrder]] = None
    error_message: Optional[str] = None
    build_time_ms: Optional[float] = None
    
    @property
    def bid_value(self) -> int:
        """Get the bid value, returning 0 if block building failed."""
        if self.block_trace:
            return self.block_trace.bid_value
        return 0
    
    @property
    def total_gas_used(self) -> int:
        """Get total gas used."""
        if self.block_trace:
            return self.block_trace.gas_used
        return 0
    
    @property
    def profit_per_gas(self) -> float:
        """Calculate profit per gas used."""
        if self.block_trace and self.block_trace.gas_used > 0:
            return self.block_trace.bid_value / self.block_trace.gas_used
        return 0.0


class BuilderComparison:
    """
    Utility class for comparing builder results and selecting the best one.
    """
    
    @staticmethod
    def select_best_builder(results: List[BlockResult]) -> Optional[BlockResult]:
        """
        Select the best builder result based on bid value.
        
        Args:
            results: List of builder results to compare
            
        Returns:
            The builder result with the highest bid value, or None if no successful builds
        """
        successful_results = [r for r in results if r.success and r.block_trace]
        
        if not successful_results:
            return None
        
        # Sort by bid value (highest first)
        successful_results.sort(key=lambda r: r.bid_value, reverse=True)
        return successful_results[0]
    
    @staticmethod
    def print_comparison(results: List[BlockResult]) -> None:
        """
        Print a comparison of all builder results.
        
        Args:
            results: List of builder results to display
        """
        print("\n" + "="*80)
        print("BUILDER COMPARISON RESULTS")
        print("="*80)
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        if successful_results:
            print(f"\nSuccessful builders ({len(successful_results)}):")
            print("-" * 50)
            
            # Sort by bid value
            successful_results.sort(key=lambda r: r.bid_value, reverse=True)
            
            for i, result in enumerate(successful_results, 1):
                print(f"{i}. {result.builder_name}")
                print(f"   True Bid Value: {result.bid_value / 10**18:.6f} ETH")
                if hasattr(result.block_trace, 'raw_coinbase_profit') and result.block_trace.raw_coinbase_profit:
                    print(f"   Raw Coinbase Profit: {result.block_trace.raw_coinbase_profit / 10**18:.6f} ETH")
                    if hasattr(result.block_trace, 'payout_gas_cost') and result.block_trace.payout_gas_cost:
                        print(f"   Payout Gas Cost: {result.block_trace.payout_gas_cost / 10**18:.6f} ETH")
                print(f"   Gas Used: {result.total_gas_used:,}")
                print(f"   Orders: {result.block_trace.num_orders if result.block_trace else 0}")
                print(f"   Build Time: {result.build_time_ms:.2f}ms")
                print(f"   Profit/Gas: {result.profit_per_gas:.2f} wei/gas")
                print()
        
        if failed_results:
            print(f"Failed builders ({len(failed_results)}):")
            print("-" * 30)
            for result in failed_results:
                print(f"- {result.builder_name}: {result.error_message}")
        
        # Show winner
        best = BuilderComparison.select_best_builder(results)
        if best:
            print(f"🏆 WINNER: {best.builder_name} with {best.bid_value / 10**18:.6f} ETH")
        else:
            print("❌ No successful builders")
        
        print("="*80)
    
    @staticmethod
    def print_winning_builder_details(results: List[BlockResult]) -> None:
        """
        Print detailed results for the winning builder in rbuilder style.
        
        Args:
            results: List of builder results
        """
        best = BuilderComparison.select_best_builder(results)
        if not best:
            print("❌ No successful builders")
            return
        
        print("\n" + "="*80)
        print(f"WINNING BUILDER: {best.builder_name.upper()}")
        print("="*80)
        
        print(f"True builder profit: {best.bid_value / 10**18:.6f} ETH")
        if hasattr(best.block_trace, 'raw_coinbase_profit') and best.block_trace.raw_coinbase_profit:
            print(f"Raw coinbase profit: {best.block_trace.raw_coinbase_profit / 10**18:.6f} ETH")
            if hasattr(best.block_trace, 'payout_gas_cost') and best.block_trace.payout_gas_cost:
                print(f"Payout gas cost: {best.block_trace.payout_gas_cost / 10**18:.6f} ETH")
        print(f"Number of used orders: {len(best.included_orders)}")
        print(f"Gas used: {best.total_gas_used:,}")
        print(f"Build time: {best.build_time_ms:.2f}ms")
        
        if best.included_orders:
            print("\nUsed orders:")
            # Show orders in the order they were actually included (not sorted by profit)
            
            for order_result in best.included_orders:
                order_id = order_result.order.id()
                gas_used = order_result.sim_value.gas_used
                profit_eth = order_result.sim_value.coinbase_profit / 10**18

                print(f"        {str(order_id)} gas: {gas_used:>8} profit: {profit_eth:.18f}")

        print("="*80)
