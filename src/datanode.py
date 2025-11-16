import copy
import textwrap
from node import Node


class DataNode(Node):
    """
    DataNode processes structured data and outputs transformed data.
    Takes data dict input, performs transformations, returns modified data dict.
    Only supports custom operations.
    """

    def __init__(self, node_id=None, operations=None, input_schema=None):
        super().__init__(node_id, input_schema)
        self.operations = operations or []
        self._validate_operations()
    
    def _validate_operations(self):
        """Validate operations list."""
        if not self.operations:
            raise ValueError("DataNode requires at least one operation")
        
        for op in self.operations:
            if 'type' not in op:
                raise ValueError(f"Operation missing 'type': {op}")
            if op['type'] != 'custom':
                raise ValueError(f"DataNode only supports 'custom' operations, got: {op['type']}")
    
    def read_inputs(self):
        """Read inputs with deep copy for data dicts."""
        results = {}
        for input_name, (node, buffer_key) in self.inputs.items():
            if node.readable:
                data = node._safe_read(buffer_key)
                
                # Deep copy for data dicts
                if isinstance(data, dict):
                    results[input_name] = copy.deepcopy(data)
                else:
                    results[input_name] = data
        
        return results
    
    def process(self, inputs):
        """Process data through custom operation."""
        if not inputs:
            return {}
        
        # Get first data dict as primary input
        data = None
        for value in inputs.values():
            if isinstance(value, dict):
                data = value
                break
        
        if data is None:
            return {}
        
        # Process only the first operation
        if self.operations:
            return self._apply_operation(data, self.operations[0], inputs)
        
        return {}
    
    def _apply_operation(self, data, op, inputs):
        """Apply custom operation on data."""
        op_type = op['type']
        params = op.get('params', {})
        
        if op_type != 'custom':
            print(f"[ERROR] DataNode only supports 'custom' operations")
            return data
        
        code = params.get('code', '')
        
        if not code.strip():
            print("[WARNING] Custom operation has no code")
            return data
        
        # Wrap user code in function
        func_code = f"""
def custom_operation(data, inputs):
{textwrap.indent(code, '    ')}
"""
        
        namespace = {}
        
        try:
            exec(func_code, namespace)
            custom_func = namespace['custom_operation']
            result = custom_func(data, inputs)
            
            # Return result if it's a dict, otherwise return original
            return result if isinstance(result, dict) else data
            
        except Exception as e:
            print(f"[ERROR] Custom data operation failed: {e}")
            return data
    
    def _write_output(self, data):
        """Write processed data to buffer."""
        with self._lock:
            if data is not self._last_processed_data:
                self.write_buffer = data
                self._last_processed_data = data