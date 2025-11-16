import cv2
from node import Node


class CameraNode(Node):
    """
    Camera capture node - reads from webcam or video device.
    Outputs raw frames to write_buffer.
    """

    def __init__(self, node_id=None, camera_id=0, input_schema=None):
        super().__init__(node_id, input_schema)
        self.camera_id = camera_id
        self.camera = cv2.VideoCapture(camera_id)
        
        if not self.camera.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

    def process(self, inputs):
        """Capture frame from camera (ignores inputs)."""
        ret, frame = self.camera.read()
        if ret:
            return frame
        return None

    def release(self):
        """Release camera resources."""
        if self.camera is not None:
            self.camera.release()