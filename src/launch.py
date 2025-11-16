import cv2
import time
from graph import Graph
from imagenode import ImageNode
from cameranode import CameraNode
from extractnode import ExtractNode
from datanode import DataNode
from yolonode import YOLONode


def main():
    # Initialize graph
    graph = Graph()
    
    # Register node types
    graph.register_node_type('ImageNode', ImageNode)
    graph.register_node_type('CameraNode', CameraNode)
    graph.register_node_type('ExtractNode', ExtractNode)
    graph.register_node_type('DataNode', DataNode)
    graph.register_node_type('YOLONode', YOLONode)
    
    # Load configuration
    print("Loading graph configuration...")
    graph.load_config('configs/config.yaml')
    
    # Print execution order
    print("\n=== Graph Execution Order ===")
    status = graph.get_status()
    for pool_id, pool_info in status['pools'].items():
        print(f"\n{pool_id} (FPS: {pool_info['fps']}):")
        print(f"  Execution order: {' → '.join(pool_info['sorted_order'])}")
    print("\n=============================\n")
    
    # Start graph execution
    graph.start()
    print("Graph started!")
    print("YOLO detection running at 20 FPS")
    print("Visualization running at 60 FPS")
    print("Press 'q' to quit.\n")
    
    # Get visualizer node for display
    visualizer_node = graph.get_node('visualizer')
    yolo_node = graph.get_node('yolo_detector')
    
    frame_count = 0
    start_time = time.time()
    last_print = time.time()
    
    try:
        while True:
            # Display visualizer output
            frame = visualizer_node._safe_read()
            if frame is not None:
                cv2.imshow('YOLO Detection', frame)
                frame_count += 1
            
            # Print stats every second
            current_time = time.time()
            if current_time - last_print >= 1.0:
                elapsed = current_time - start_time
                actual_fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Get detection count
                yolo_data = yolo_node._safe_read()
                det_count = 0
                if yolo_data and 'output0' in yolo_data:
                    det_count = yolo_data['output0'].get('count', 0)
                
                print(f"Display FPS: {actual_fps:.1f} | Detections: {det_count}")
                last_print = current_time
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.001)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        print("\nStopping graph...")
        graph.stop()
        
        camera_node = graph.get_node('camera')
        if hasattr(camera_node, 'release'):
            camera_node.release()
        
        cv2.destroyAllWindows()
        
        # Print final statistics
        elapsed = time.time() - start_time
        print(f"\n=== Session Statistics ===")
        print(f"Runtime: {elapsed:.1f}s")
        print(f"Total frames displayed: {frame_count}")
        print(f"Average display FPS: {frame_count / elapsed:.1f}")
        
        status = graph.get_status()
        for node_id in ['camera', 'yolo_detector', 'visualizer']:
            node_status = status['nodes'].get(node_id, {})
            exec_count = node_status.get('execution_count', 0)
            print(f"{node_id}: {exec_count} executions")
        print("==========================\n")
        
        print("Done! 🎉")


if __name__ == '__main__':
    main()