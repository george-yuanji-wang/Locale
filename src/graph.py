import threading
import yaml
import time
from typing import Dict, List, Type
from node import Node


class ThreadPool:
    """Manages a single thread executing multiple nodes in topological order."""

    def __init__(self, pool_id: str, fps: float = None):
        self.id = pool_id
        self.fps = fps
        self.nodes: List[Node] = []
        self.sorted_nodes: List[Node] = []
        self.thread = None
        self.running = False
        
        self._tick_interval = (1.0 / fps) if fps else 0.0
        self._last_tick = 0.0

    def topological_sort(self):
        """
        Sort nodes in execution order using Kahn's algorithm.
        Only considers dependencies within this pool.
        """
        # Build dependency graph
        in_degree = {node: 0 for node in self.nodes}
        adjacency = {node: [] for node in self.nodes}
        
        for node in self.nodes:
            for input_name, (input_node, _) in node.inputs.items():
                # Only count dependencies within this pool
                if input_node in self.nodes:
                    in_degree[node] += 1
                    adjacency[input_node].append(node)
        
        # Initialize queue with zero in-degree nodes
        queue = [node for node in self.nodes if in_degree[node] == 0]
        queue.sort(key=lambda n: n.id)  # Deterministic ordering
        
        sorted_list = []
        
        while queue:
            # Pop first node
            current = queue.pop(0)
            sorted_list.append(current)
            
            # Process downstream nodes
            for downstream in adjacency[current]:
                in_degree[downstream] -= 1
                if in_degree[downstream] == 0:
                    queue.append(downstream)
                    queue.sort(key=lambda n: n.id)  # Maintain deterministic order
        
        # Check for cycles
        if len(sorted_list) != len(self.nodes):
            remaining = [node.id for node in self.nodes if node not in sorted_list]
            raise ValueError(f"Cycle detected in pool '{self.id}'. Remaining nodes: {remaining}")
        
        self.sorted_nodes = sorted_list

    def should_tick(self):
        """Check if pool should execute this iteration based on FPS."""
        if self.fps is None:
            return True
        
        current_time = time.time()
        elapsed = current_time - self._last_tick
        if elapsed >= self._tick_interval:
            self._last_tick = current_time
            return True
        return False

    def run(self):
        """Main pool execution loop."""
        while self.running:
            if self.should_tick():
                for node in self.sorted_nodes:
                    try:
                        node.run()
                    except Exception as e:
                        print(f"[ERROR] Node {node.id} in pool {self.id}: {e}")
            
            time.sleep(0.001)  # Minimal sleep to prevent CPU spinning

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self, timeout=5.0):
        self.running = False
        if self.thread:
            self.thread.join(timeout=timeout)


class Graph:
    """Orchestrates node network execution via thread pools."""

    def __init__(self):
        self.pools: Dict[str, ThreadPool] = {}
        self.nodes: Dict[str, Node] = {}
        self.node_types: Dict[str, Type[Node]] = {}

    def register_node_type(self, type_name: str, node_class: Type[Node]):
        """Register custom node class for factory instantiation."""
        self.node_types[type_name] = node_class

    def _get_output_type(self, node):
        """Infer what type this node outputs."""
        # Import here to avoid circular dependencies
        from cameranode import CameraNode
        from imagenode import ImageNode
        from extractnode import ExtractNode
        from datanode import DataNode
        
        if isinstance(node, (CameraNode, ImageNode)):
            return 'image'
        elif isinstance(node, (ExtractNode, DataNode)):
            return 'data'
        return 'unknown'

    def _get_input_type(self, node, input_name):
        """Get expected type from node's input schema."""
        if not node.input_schema:
            return 'any'
        
        for inp in node.input_schema:
            if inp.get('name') == input_name:
                return inp.get('type', 'any')
        
        return 'unknown'

    def _types_compatible(self, expected, actual):
        """Check if output type can connect to input type."""
        if expected == 'any' or actual == 'unknown':
            return True
        return expected == actual

    def validate_connections(self):
        """Validate all connections are type-compatible before execution."""
        errors = []
        
        for node_id, node in self.nodes.items():
            for input_name, (source_node, _) in node.inputs.items():
                # Check 1: If node has schema, verify input name exists
                if node.input_schema:
                    schema_input_names = [inp.get('name') for inp in node.input_schema]
                    if input_name not in schema_input_names:
                        errors.append(
                            f"Node '{node_id}' does not declare input '{input_name}'. "
                            f"Schema expects: {schema_input_names}"
                        )
                        continue
                
                # Check 2: Type compatibility
                expected_type = self._get_input_type(node, input_name)
                actual_type = self._get_output_type(source_node)
                
                if not self._types_compatible(expected_type, actual_type):
                    errors.append(
                        f"Type mismatch: '{source_node.id}' (outputs {actual_type}) → "
                        f"'{node_id}.{input_name}' (expects {expected_type})"
                    )
        
        if errors:
            error_msg = "Graph validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)

    def load_config(self, config_path: str):
        """Parse YAML config and build graph topology."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Create pools
        for pool_config in config.get('thread_pools', []):
            pool_id = pool_config['id']
            fps = pool_config.get('fps')
            self.pools[pool_id] = ThreadPool(pool_id, fps)

        # Create nodes
        for node_config in config.get('nodes', []):
            node_id = node_config['id']
            node_type = node_config['type']
            pool_id = node_config['pool']
            params = node_config.get('params', {})
            input_schema = node_config.get('input_schema', [])
            
            # Handle operations for ImageNode/ExtractNode/DataNode
            if 'operations' in node_config:
                params['operations'] = node_config['operations']

            if node_type not in self.node_types:
                raise ValueError(f"Unknown node type: {node_type}")

            node_class = self.node_types[node_type]
            node = node_class(node_id=node_id, input_schema=input_schema, **params)
            
            self.nodes[node_id] = node
            self.pools[pool_id].nodes.append(node)

        # Create connections
        for conn in config.get('connections', []):
            source_id = conn['source']
            target_id = conn['target']
            buffer_key = conn.get('source_buffer')
            target_input = conn.get('target_input', source_id)  # Default: use source node id as input name

            source_node = self.nodes[source_id]
            target_node = self.nodes[target_id]
            target_node.bind_input(target_input, source_node, buffer_key)
        
        # Validate connections before execution
        self.validate_connections()
        
        # Perform topological sort for each pool
        for pool in self.pools.values():
            pool.topological_sort()

    def start(self):
        """Start all thread pools."""
        for pool in self.pools.values():
            pool.start()

    def stop(self):
        """Stop all thread pools gracefully."""
        for pool in self.pools.values():
            pool.stop()

    def get_node(self, node_id: str) -> Node:
        """Retrieve node by ID."""
        return self.nodes.get(node_id)

    def get_status(self) -> dict:
        """Get graph-wide status."""
        return {
            'pools': {pid: {
                'nodes': len(p.nodes), 
                'sorted_order': [n.id for n in p.sorted_nodes],
                'running': p.running,
                'fps': p.fps
            } for pid, p in self.pools.items()},
            'nodes': {nid: n.get_status() for nid, n in self.nodes.items()}
        }