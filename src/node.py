import uuid
import threading
import time


class Node:
    """Base Node class for data flow processing with reference-based linking."""

    def __init__(self, node_id: str = None, input_schema: list = None):
        self.id = node_id or self._generate_id()
        self.input_schema = input_schema or []
        self.inputs = {}  # Changed to dict: {input_name: (node, buffer_key)}
        self.write_buffer = None
        self.readable = True
        self._last_processed_data = None
        self._last_valid_output = None  # NEW: For fallback on failure
        self._lock = threading.Lock()
        
        self._execution_count = 0
        self._last_execution_duration = 0.0

    def _generate_id(self) -> str:
        return str(uuid.uuid4())[:8]

    def bind_input(self, input_name: str, other_node: "Node", buffer_key: str = None):
        """Bind named input to source node."""
        if not other_node.readable:
            raise ValueError(f"Node {other_node.id} is not readable")
        self.inputs[input_name] = (other_node, buffer_key)

    def read_inputs(self):
        """Returns dict of {input_name: data}."""
        results = {}
        for input_name, (node, buffer_key) in self.inputs.items():
            if node.readable:
                data = node._safe_read(buffer_key)
                results[input_name] = data
        return results

    def _safe_read(self, buffer_key=None):
        with self._lock:
            return self.write_buffer

    def process(self, inputs):
        """
        Process inputs (dict of named inputs).
        Override in child classes for specific behavior.
        """
        # Default: return first available data
        return next(iter(inputs.values())) if inputs else None

    def run(self):
        start = time.perf_counter()
        
        try:
            input_data = self.read_inputs()
            processed = self.process(input_data)
            
            # Fallback logic: if processing returns None/empty and we have a fallback
            if processed is None or (isinstance(processed, dict) and not processed):
                if self._last_valid_output is not None:
                    print(f"[FALLBACK] Node {self.id} using last valid output")
                    processed = self._last_valid_output
            
            self._write_output(processed)
            
        except Exception as e:
            print(f"[ERROR] Node {self.id} failed: {e}")
            # Use fallback on exception
            if self._last_valid_output is not None:
                print(f"[FALLBACK] Node {self.id} recovering with last valid output")
                self._write_output(self._last_valid_output)
        
        self._last_execution_duration = time.perf_counter() - start
        self._execution_count += 1

    def _write_output(self, data):
        with self._lock:
            # Track valid outputs for fallback
            if data is not None and (not isinstance(data, dict) or data):
                self._last_valid_output = data
            
            if data is not self._last_processed_data:
                self.write_buffer = data
                self._last_processed_data = data

    def get_status(self) -> dict:
        return {
            'id': self.id,
            'inputs': len(self.inputs),
            'input_schema': self.input_schema,
            'readable': self.readable,
            'buffer_state': self.write_buffer is not None,
            'execution_count': self._execution_count,
            'last_duration': self._last_execution_duration
        }

    def set_readable(self, state: bool):
        self.readable = state