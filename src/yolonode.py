import os
import torch
import numpy as np
from pathlib import Path
from node import Node


class YOLONode(Node):
    """
    YOLO object detection node with batched inference.
    Processes multiple image inputs simultaneously for efficiency.
    Supports GPU (CUDA/MPS) and CPU execution.
    """
    
    def __init__(self, node_id=None, model_name='yolov8s', device='auto',
                 num_inputs=4, confidence=0.5, iou=0.45, input_schema=None):
        
        # Generate input schema dynamically if not provided
        if input_schema is None:
            input_schema = [
                {'name': f'input{i}', 'type': 'image'}
                for i in range(num_inputs)
            ]
        
        super().__init__(node_id, input_schema=input_schema)
        
        self.model_name = model_name
        self.num_inputs = num_inputs
        self.confidence = confidence
        self.iou = iou
        self.device = self._resolve_device(device)
        
        self.model = self._load_model()
    
    def _resolve_device(self, device_config):
        """Determine best available device."""
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
            print(f"[YOLO] Auto-selected device: {device}")
            return device
        else:
            print(f"[YOLO] Using specified device: {device_config}")
            return device_config
    
    def _find_model_path(self, model_name):
        """Find model file in models/yolo directory."""
        local_path = Path('models/yolo') / f"{model_name}.pt"
        
        if local_path.exists():
            print(f"[YOLO] Found model: {local_path}")
            return str(local_path)
        
        # Not found - fail with helpful message
        raise FileNotFoundError(
            f"\nYOLO model '{model_name}.pt' not found!\n\n"
            f"Expected location: models/yolo/{model_name}.pt\n\n"
            f"Please download the model:\n"
            f"  1. Visit: https://github.com/ultralytics/assets/releases\n"
            f"  2. Download {model_name}.pt\n"
            f"  3. Place in: models/yolo/{model_name}.pt\n\n"
            f"Or use the helper script:\n"
            f"  cd models/yolo && bash download.sh\n\n"
            f"Available models: yolov8n, yolov8s, yolov8m\n"
        )
    
    def _load_model(self):
        """Load YOLO model from local path."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not installed.\n"
                "Install with: pip install ultralytics torch torchvision"
            )
        
        model_path = self._find_model_path(self.model_name)
        model = YOLO(model_path)
        
        # Move to device
        if self.device != 'cpu':
            try:
                model.to(self.device)
                print(f"[YOLO] Node {self.id}: {self.model_name} loaded on {self.device}")
            except Exception as e:
                print(f"[YOLO] Warning: Failed to move model to {self.device}: {e}")
                print(f"[YOLO] Falling back to CPU")
                self.device = 'cpu'
        else:
            print(f"[YOLO] Node {self.id}: {self.model_name} loaded on CPU")
        
        return model
    
    def read_inputs(self):
        """Read image inputs - no defensive copy (YOLO doesn't modify)."""
        results = {}
        for input_name, (node, buffer_key) in self.inputs.items():
            if node.readable:
                data = node._safe_read(buffer_key)
                results[input_name] = data
        return results
    
    def process(self, inputs):
        """Run batched YOLO inference on all input images."""
        # Collect images in order: input0, input1, input2, ...
        images = []
        input_keys = []
        
        for i in range(self.num_inputs):
            input_key = f'input{i}'
            if input_key in inputs and isinstance(inputs[input_key], np.ndarray):
                images.append(inputs[input_key])
                input_keys.append(input_key)
        
        if not images:
            # No valid images, return empty results for all inputs
            return {
                f'output{i}': {'detections': [], 'count': 0}
                for i in range(self.num_inputs)
            }
        
        # Batch inference (all images processed together)
        try:
            results = self.model(
                images,
                conf=self.confidence,
                iou=self.iou,
                verbose=False
            )
        except Exception as e:
            print(f"[YOLO] Inference failed: {e}")
            return {
                f'output{i}': {'detections': [], 'count': 0}
                for i in range(self.num_inputs)
            }
        
        # Format output for each input
        output = {}
        for i, (input_key, result) in enumerate(zip(input_keys, results)):
            input_index = int(input_key.replace('input', ''))
            output[f'output{input_index}'] = self._format_detections(result)
        
        # Fill in empty results for inputs that weren't processed
        for i in range(self.num_inputs):
            key = f'output{i}'
            if key not in output:
                output[key] = {'detections': [], 'count': 0}
        
        return output
    
    def _format_detections(self, result):
        """Format YOLO results into standard data structure."""
        detections = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = box
                detections.append({
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2),
                        'width': float(x2 - x1),
                        'height': float(y2 - y1),
                        'center_x': float((x1 + x2) / 2),
                        'center_y': float((y1 + y2) / 2)
                    },
                    'class_id': int(cls),
                    'class_name': result.names[int(cls)],
                    'confidence': float(conf)
                })
        
        return {
            'detections': detections,
            'count': len(detections)
        }
    
    def _write_output(self, data):
        """Write detection data to buffer."""
        with self._lock:
            # Track valid outputs for fallback
            if data is not None and (not isinstance(data, dict) or data):
                self._last_valid_output = data
            
            if data is not self._last_processed_data:
                self.write_buffer = data
                self._last_processed_data = data