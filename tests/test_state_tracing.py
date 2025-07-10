import unittest
from unittest.mock import Mock, patch
from backtest.build.simulation.state_trace import UsedStateTrace, SlotKey
from backtest.build.simulation.state_tracer import StateTracker, TracingEVMWrapper


class TestStateTrace(unittest.TestCase):
    """Test the UsedStateTrace functionality"""
    
    def test_slot_key_equality(self):
        """Test SlotKey equality and hashing"""
        key1 = SlotKey("0x123", 1)
        key2 = SlotKey("0x123", 1)
        key3 = SlotKey("0x123", 2)
        
        self.assertEqual(key1, key2)
        self.assertNotEqual(key1, key3)
        self.assertEqual(hash(key1), hash(key2))
        self.assertNotEqual(hash(key1), hash(key3))
    
    def test_trace_merge(self):
        """Test merging of state traces"""
        trace1 = UsedStateTrace()
        trace1.read_slot_values[SlotKey("0x123", 0)] = 100
        trace1.written_slot_values[SlotKey("0x123", 1)] = 200
        trace1.sent_amount["0x456"] = 1000
        
        trace2 = UsedStateTrace()
        trace2.read_slot_values[SlotKey("0x123", 2)] = 300
        trace2.written_slot_values[SlotKey("0x123", 1)] = 400  # Override
        trace2.received_amount["0x789"] = 500
        
        merged = trace1.merge(trace2)
        
        # Check storage merging
        self.assertEqual(len(merged.read_slot_values), 2)
        self.assertEqual(merged.read_slot_values[SlotKey("0x123", 0)], 100)
        self.assertEqual(merged.read_slot_values[SlotKey("0x123", 2)], 300)
        
        # Check write override (last write wins)
        self.assertEqual(merged.written_slot_values[SlotKey("0x123", 1)], 400)
        
        # Check amount merging
        self.assertEqual(merged.sent_amount["0x456"], 1000)
        self.assertEqual(merged.received_amount["0x789"], 500)
    
    def test_conflict_detection(self):
        """Test conflict detection between traces"""
        trace1 = UsedStateTrace()
        trace1.written_slot_values[SlotKey("0x123", 0)] = 100
        trace1.sent_amount["0x456"] = 1000
        
        # Conflicting trace (same storage slot)
        trace2 = UsedStateTrace()
        trace2.written_slot_values[SlotKey("0x123", 0)] = 200
        
        # Non-conflicting trace (different storage slot)
        trace3 = UsedStateTrace()
        trace3.written_slot_values[SlotKey("0x123", 1)] = 300
        
        self.assertTrue(trace1.conflicts_with(trace2))
        self.assertFalse(trace1.conflicts_with(trace3))
    
    def test_gas_estimation(self):
        """Test gas estimation from access patterns"""
        trace = UsedStateTrace()
        
        # Add 2 addresses and 3 storage slots
        trace.accessed_addresses = {"0x123", "0x456"}
        trace.accessed_storage_keys = {
            SlotKey("0x123", 0),
            SlotKey("0x123", 1),
            SlotKey("0x456", 0)
        }
        
        estimated_gas = trace.get_total_gas_estimate()
        expected_gas = 2 * 2600 + 3 * 2100  # 2 accounts + 3 storage slots
        self.assertEqual(estimated_gas, expected_gas)
    
    def test_summary(self):
        """Test trace summary generation"""
        trace = UsedStateTrace()
        trace.read_slot_values[SlotKey("0x123", 0)] = 100
        trace.written_slot_values[SlotKey("0x123", 1)] = 200
        trace.created_contracts = ["0x789"]
        
        summary = trace.summary()
        
        self.assertEqual(summary["storage_reads"], 1)
        self.assertEqual(summary["storage_writes"], 1)
        self.assertEqual(summary["contracts_created"], 1)
        self.assertIn("estimated_gas_overhead", summary)


class TestStateTracker(unittest.TestCase):
    """Test the StateTracker functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_evm = Mock()
        self.mock_evm.journal_state = {}
        self.mock_evm.storage.return_value = 0
        self.tracker = StateTracker(self.mock_evm)
    
    def test_start_tracking(self):
        """Test starting state tracking"""
        self.mock_evm.journal_state = {"0x123": Mock()}
        
        self.tracker.start_tracking()
        
        self.assertIsNotNone(self.tracker.pre_state)
        self.assertEqual(len(self.tracker.pre_state), 1)
        self.assertIsInstance(self.tracker.trace, UsedStateTrace)
    
    def test_storage_access_tracking(self):
        """Test manual storage access tracking"""
        self.tracker.start_tracking()
        
        # Track a storage read
        self.tracker.track_storage_access("0x123", 0, 100, is_write=False)
        
        slot_key = SlotKey("0x123", 0)
        self.assertIn(slot_key, self.tracker.trace.read_slot_values)
        self.assertEqual(self.tracker.trace.read_slot_values[slot_key], 100)
        self.assertIn(slot_key, self.tracker.trace.accessed_storage_keys)
        
        # Track a storage write
        self.tracker.track_storage_access("0x123", 0, 200, is_write=True)
        
        self.assertIn(slot_key, self.tracker.trace.written_slot_values)
        self.assertEqual(self.tracker.trace.written_slot_values[slot_key], 200)
    
    def test_balance_access_tracking(self):
        """Test balance access tracking"""
        self.tracker.start_tracking()
        
        # Track balance access
        self.tracker.track_balance_access("0x123", 1000)
        
        self.assertIn("0x123", self.tracker.trace.read_balances)
        self.assertEqual(self.tracker.trace.read_balances["0x123"], 1000)
        self.assertIn("0x123", self.tracker.trace.accessed_addresses)


class TestTracingEVMWrapper(unittest.TestCase):
    """Test the TracingEVMWrapper functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_evm = Mock()
        self.mock_evm.tracing = False
        self.mock_evm.journal_state = {}  # Make it a real dict instead of Mock
        self.wrapper = TracingEVMWrapper(self.mock_evm, auto_trace=True)
    
    def test_tracing_enablement(self):
        """Test that tracing is enabled on wrapper creation"""
        # The wrapper should enable tracing if not already enabled
        self.mock_evm.set_tracing = True
    
    def test_start_tracing(self):
        """Test manual tracing start"""
        tracker = self.wrapper.start_tracing()
        
        self.assertIsNotNone(self.wrapper.tracker)
        self.assertIsInstance(tracker, StateTracker)
    
    def test_attribute_delegation(self):
        """Test that attributes are delegated to the wrapped EVM"""
        self.mock_evm.some_method.return_value = "test_result"
        
        result = self.wrapper.some_method()
        
        self.assertEqual(result, "test_result")
        self.mock_evm.some_method.assert_called_once()
    
    def test_message_call_with_auto_trace(self):
        """Test message_call with automatic tracing"""
        self.mock_evm.message_call.return_value = b"result"
        
        # Call should start tracing automatically
        result = self.wrapper.message_call(caller="0x123", to="0x456")
        
        self.assertEqual(result, b"result")
        self.mock_evm.message_call.assert_called_once()
        self.assertIsNotNone(self.wrapper.tracker)


if __name__ == "__main__":
    unittest.main()
