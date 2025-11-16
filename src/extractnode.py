import cv2
import numpy as np
import textwrap
from node import Node


class ExtractNode(Node):
    """
    ExtractNode processes images and outputs structured data.
    Takes image input, performs extraction/analysis, returns dict/structured data.
    """

    def __init__(self, node_id=None, operations=None, input_schema=None):
        super().__init__(node_id, input_schema)
        self.operations = operations or []
        self._validate_operations()
    
    def _validate_operations(self):
        """Validate operations list."""
        if not self.operations:
            raise ValueError("ExtractNode requires at least one operation")
        
        for op in self.operations:
            if 'type' not in op:
                raise ValueError(f"Operation missing 'type': {op}")
    
    def read_inputs(self):
        """Read inputs - no defensive copy needed (ExtractNode doesn't modify)."""
        results = {}
        for input_name, (node, buffer_key) in self.inputs.items():
            if node.readable:
                data = node._safe_read(buffer_key)
                results[input_name] = data
        return results
    
    def process(self, inputs):
        """Extract data from image using first operation."""
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
        
        # Process only the first operation
        if self.operations:
            return self._apply_operation(img, self.operations[0], inputs)
        
        return None
    
    def _apply_operation(self, img, op, inputs):
        """Apply extraction operation and return structured data."""
        op_type = op['type']
        params = op.get('params', {})
        
        try:
            # Contour detection
            if op_type == 'find_contours':
                mode = getattr(cv2, params.get('mode', 'RETR_EXTERNAL'))
                method = getattr(cv2, params.get('method', 'CHAIN_APPROX_SIMPLE'))
                
                # Image must be binary/grayscale
                if len(img.shape) == 3:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    img_gray = img
                
                contours, _ = cv2.findContours(img_gray, mode, method)
                
                result = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    
                    # Optional filtering
                    min_area = params.get('min_area', 0)
                    max_area = params.get('max_area', float('inf'))
                    
                    if min_area <= area <= max_area:
                        result.append({
                            'points': contour.squeeze().tolist() if contour.size > 0 else [],
                            'area': float(area),
                            'perimeter': float(perimeter)
                        })
                
                return {'contours': result}
            
            # Bounding boxes from contours
            elif op_type == 'find_bounding_boxes':
                mode = getattr(cv2, params.get('mode', 'RETR_EXTERNAL'))
                method = getattr(cv2, params.get('method', 'CHAIN_APPROX_SIMPLE'))
                
                if len(img.shape) == 3:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    img_gray = img
                
                contours, _ = cv2.findContours(img_gray, mode, method)
                
                result = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    area = w * h
                    
                    min_area = params.get('min_area', 0)
                    max_area = params.get('max_area', float('inf'))
                    
                    if min_area <= area <= max_area:
                        result.append({
                            'x': int(x),
                            'y': int(y),
                            'w': int(w),
                            'h': int(h),
                            'area': int(area)
                        })
                
                return {'bboxes': result}
            
            # Mean color
            elif op_type == 'mean_color':
                mean = cv2.mean(img)
                if len(img.shape) == 3:
                    return {
                        'mean_color': {
                            'b': float(mean[0]),
                            'g': float(mean[1]),
                            'r': float(mean[2])
                        }
                    }
                else:
                    return {'mean_color': {'gray': float(mean[0])}}
            
            # Dominant color (simplified - mean of each channel)
            elif op_type == 'dominant_color':
                if len(img.shape) == 3:
                    # Reshape and find most common color
                    pixels = img.reshape(-1, 3)
                    mean_color = pixels.mean(axis=0)
                    return {
                        'dominant_color': {
                            'b': int(mean_color[0]),
                            'g': int(mean_color[1]),
                            'r': int(mean_color[2])
                        }
                    }
                else:
                    mean_val = img.mean()
                    return {'dominant_color': {'gray': int(mean_val)}}
            
            # Color histogram
            elif op_type == 'color_histogram':
                bins = params.get('bins', 256)
                if len(img.shape) == 3:
                    hist_b = cv2.calcHist([img], [0], None, [bins], [0, 256])
                    hist_g = cv2.calcHist([img], [1], None, [bins], [0, 256])
                    hist_r = cv2.calcHist([img], [2], None, [bins], [0, 256])
                    return {
                        'histogram': {
                            'b': hist_b.flatten().tolist(),
                            'g': hist_g.flatten().tolist(),
                            'r': hist_r.flatten().tolist()
                        }
                    }
                else:
                    hist = cv2.calcHist([img], [0], None, [bins], [0, 256])
                    return {'histogram': {'gray': hist.flatten().tolist()}}
            
            # Brightness statistics
            elif op_type == 'brightness_stats':
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img
                
                return {
                    'brightness': {
                        'mean': float(gray.mean()),
                        'std': float(gray.std()),
                        'min': int(gray.min()),
                        'max': int(gray.max())
                    }
                }
            
            # Image statistics
            elif op_type == 'image_stats':
                return {
                    'image_info': {
                        'width': int(img.shape[1]),
                        'height': int(img.shape[0]),
                        'channels': int(img.shape[2]) if len(img.shape) == 3 else 1,
                        'dtype': str(img.dtype)
                    }
                }
            
            # Count non-zero pixels
            elif op_type == 'count_nonzero':
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img
                
                count = cv2.countNonZero(gray)
                total = gray.shape[0] * gray.shape[1]
                
                return {
                    'pixel_count': {
                        'nonzero': int(count),
                        'total': int(total),
                        'percentage': float(count / total * 100)
                    }
                }
            
            # Detect corners (Harris)
            elif op_type == 'detect_corners':
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img
                
                gray = np.float32(gray)
                dst = cv2.cornerHarris(gray, 2, 3, 0.04)
                
                threshold = params.get('threshold', 0.01)
                dst_thresh = dst > threshold * dst.max()
                
                coords = np.argwhere(dst_thresh)
                corners = [{'x': int(pt[1]), 'y': int(pt[0])} for pt in coords]
                
                return {'corners': corners}
            
            # Detect blobs
            elif op_type == 'detect_blobs':
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img
                
                # Simple blob detection using threshold and contours
                _, binary = cv2.threshold(gray, params.get('threshold', 127), 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                min_area = params.get('min_area', 100)
                blobs = []
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area >= min_area:
                        M = cv2.moments(contour)
                        if M['m00'] != 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            blobs.append({
                                'x': cx,
                                'y': cy,
                                'size': float(area)
                            })
                
                return {'blobs': blobs}
            
            # Custom operation
            elif op_type == 'custom':
                code = params.get('code', '')
                output_schema = op.get('output_schema', {})
                img_copy = img.copy()
                
                if not code.strip():
                    print("[WARNING] Custom operation has no code")
                    return {}
                
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
                    
                    # No validation - assume correct (UI handles it)
                    return result if isinstance(result, dict) else {}
                    
                except Exception as e:
                    print(f"[ERROR] Custom extraction failed: {e}")
                    return {}
            
            else:
                print(f"[WARNING] Unknown extraction type: {op_type}")
                return {}
                
        except Exception as e:
            print(f"[ERROR] Extraction operation {op_type} failed: {e}")
            return {}
    
    def _write_output(self, data):
        """Write extracted data to buffer."""
        with self._lock:
            if data is not self._last_processed_data:
                self.write_buffer = data
                self._last_processed_data = data