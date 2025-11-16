import cv2
import numpy as np
import textwrap
from node import Node


class ImageNode(Node):
    """
    Configurable ImageNode supporting common CV operations via config.
    Pure processing node - does not capture from camera.
    """

    def __init__(self, node_id=None, operations=None, input_schema=None):
        super().__init__(node_id, input_schema)
        self.operations = operations or []
        self._validate_operations()
    
    def _validate_operations(self):
        """Validate all operations have required parameters."""
        for op in self.operations:
            if 'type' not in op:
                raise ValueError(f"Operation missing 'type': {op}")

    def read_inputs(self):
        """Read inputs - handles both image and data types with defensive copying."""
        results = {}
        for input_name, (node, buffer_key) in self.inputs.items():
            if node.readable:
                data = node._safe_read(buffer_key)
                
                # Defensive copy for images only
                if isinstance(data, np.ndarray):
                    results[input_name] = data.copy()
                else:
                    # Data dicts, no copy needed
                    results[input_name] = data
        
        return results

    def process(self, inputs):
        """Process images through configured operation pipeline."""
        if not inputs:
            return None
        
        # Get first image input
        img = None
        for value in inputs.values():
            if isinstance(value, np.ndarray):
                img = value
                break
        
        if img is None:
            return None
        
        # Apply operation pipeline
        for op in self.operations:
            img = self._apply_operation(img, op, inputs)
            if img is None:
                break
        
        return img

    def _apply_operation(self, img, op, inputs):
        """Apply single CV operation based on type."""
        op_type = op['type']
        params = op.get('params', {})
        
        try:
            # Color conversions
            if op_type == 'cvtColor':
                code = getattr(cv2, params['code'])
                return cv2.cvtColor(img, code)
            
            # Blurring operations
            elif op_type == 'GaussianBlur':
                ksize = tuple(params['ksize'])
                return cv2.GaussianBlur(img, ksize, params.get('sigmaX', 0))
            
            elif op_type == 'medianBlur':
                return cv2.medianBlur(img, params['ksize'])
            
            elif op_type == 'bilateralFilter':
                return cv2.bilateralFilter(img, params['d'], params['sigmaColor'], params['sigmaSpace'])
            
            # Edge detection
            elif op_type == 'Canny':
                return cv2.Canny(img, params['threshold1'], params['threshold2'])
            
            elif op_type == 'Sobel':
                dx = params.get('dx', 1)
                dy = params.get('dy', 0)
                ksize = params.get('ksize', 3)
                return cv2.Sobel(img, cv2.CV_64F, dx, dy, ksize=ksize)
            
            elif op_type == 'Laplacian':
                ksize = params.get('ksize', 1)
                return cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
            
            # Thresholding
            elif op_type == 'threshold':
                _, thresh = cv2.threshold(img, params['thresh'], params['maxval'], 
                                         getattr(cv2, params['type']))
                return thresh
            
            elif op_type == 'adaptiveThreshold':
                return cv2.adaptiveThreshold(
                    img, params['maxval'],
                    getattr(cv2, params['adaptiveMethod']),
                    getattr(cv2, params['thresholdType']),
                    params['blockSize'], params['C']
                )
            
            # Geometric transforms
            elif op_type == 'resize':
                size = tuple(params['size'])
                interpolation = getattr(cv2, params.get('interpolation', 'INTER_LINEAR'))
                return cv2.resize(img, size, interpolation=interpolation)
            
            elif op_type == 'flip':
                return cv2.flip(img, params['flipCode'])
            
            elif op_type == 'rotate':
                angle = params['angle']
                if angle == 90:
                    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    return cv2.rotate(img, cv2.ROTATE_180)
                elif angle == 270:
                    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    # Arbitrary angle rotation
                    h, w = img.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    return cv2.warpAffine(img, M, (w, h))
            
            elif op_type == 'crop':
                x, y, w, h = params['x'], params['y'], params['w'], params['h']
                return img[y:y+h, x:x+w]
            
            elif op_type == 'warpAffine':
                M = np.array(params['matrix'], dtype=np.float32)
                size = tuple(params['size'])
                return cv2.warpAffine(img, M, size)
            
            elif op_type == 'warpPerspective':
                M = np.array(params['matrix'], dtype=np.float32)
                size = tuple(params['size'])
                return cv2.warpPerspective(img, M, size)
            
            # Morphological operations
            elif op_type == 'erode':
                kernel = np.ones(tuple(params['kernel']), np.uint8)
                return cv2.erode(img, kernel, iterations=params.get('iterations', 1))
            
            elif op_type == 'dilate':
                kernel = np.ones(tuple(params['kernel']), np.uint8)
                return cv2.dilate(img, kernel, iterations=params.get('iterations', 1))
            
            elif op_type == 'morphologyEx':
                kernel = np.ones(tuple(params['kernel']), np.uint8)
                op_code = getattr(cv2, params['op'])
                return cv2.morphologyEx(img, op_code, kernel)
            
            # Adjustments
            elif op_type == 'brightness':
                beta = params['value']
                return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)
            
            elif op_type == 'contrast':
                alpha = params['value']
                return cv2.convertScaleAbs(img, alpha=alpha, beta=0)
            
            elif op_type == 'equalizeHist':
                if len(img.shape) == 2:
                    return cv2.equalizeHist(img)
                else:
                    # Apply to each channel
                    channels = cv2.split(img)
                    eq_channels = [cv2.equalizeHist(ch) for ch in channels]
                    return cv2.merge(eq_channels)
            
            # Masking operations
            elif op_type == 'inRange':
                lower = np.array(params['lower'], dtype=np.uint8)
                upper = np.array(params['upper'], dtype=np.uint8)
                return cv2.inRange(img, lower, upper)
            
            elif op_type == 'bitwise_and':
                mask = self._get_mask(params.get('mask'))
                if mask is not None:
                    return cv2.bitwise_and(img, img, mask=mask)
                return img
            
            elif op_type == 'bitwise_or':
                mask = self._get_mask(params.get('mask'))
                if mask is not None:
                    return cv2.bitwise_or(img, img, mask=mask)
                return img
            
            elif op_type == 'bitwise_not':
                return cv2.bitwise_not(img)
            
            elif op_type == 'bitwise_xor':
                mask = self._get_mask(params.get('mask'))
                if mask is not None:
                    return cv2.bitwise_xor(img, img, mask=mask)
                return img
            
            # Custom operation
            elif op_type == 'custom':
                code = params.get('code', '')
                img_copy = img.copy()
                
                if not code.strip():
                    print("[WARNING] Custom operation has no code")
                    return img
                
                # Wrap user code in function
                func_code = f"""
def custom_operation(img, cv2, np, inputs):
{textwrap.indent(code, '    ')}
"""
                
                namespace = {'cv2': cv2, 'np': np}
                
                try:
                    exec(func_code, namespace)
                    custom_func = namespace['custom_operation']
                    result = custom_func(img_copy, cv2, np, inputs)
                    
                    # Validate output
                    if not isinstance(result, np.ndarray):
                        raise TypeError(f"Custom operation must return np.ndarray, got {type(result)}")
                    
                    return result
                    
                except Exception as e:
                    print(f"[ERROR] Custom operation failed: {e}")
                    return img  # Fallback to original image
            
            else:
                print(f"[WARNING] Unknown operation type: {op_type}")
                return img
                
        except Exception as e:
            print(f"[ERROR] Operation {op_type} failed: {e}")
            return img
    
    def _get_mask(self, mask_spec):
        """Helper to retrieve mask from various sources."""
        if mask_spec is None:
            return None
        
        # If mask_spec is direct array
        if isinstance(mask_spec, np.ndarray):
            return mask_spec
        
        return None

    def _write_output(self, image):
        """Write processed image to buffer."""
        if image is not None and not isinstance(image, np.ndarray):
            image = np.array(image, dtype=np.uint8)
        
        with self._lock:
            if image is not self._last_processed_data:
                self.write_buffer = image
                self._last_processed_data = image