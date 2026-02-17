#!/usr/bin/env python3
"""
Object Detection Script for Raspberry Pi Zero W with IMX500 AI Camera
Outputs detected objects as JSON to terminal
"""
from importlib.metadata import metadata
import json
from logging import config
import time
from unicodedata import category
import cv2
import numpy as np
from picamera2 import MappedArray, Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
from pymavlink import mavutil

# Detection parameters
THRESHOLD = 0.55
IOU = 0.65
MAX_DETECTIONS = 10
HORIZONTAL_FOV_DEG = 60.0       #Adjust to your actual IMX500 horizontal FOV

# Streaming parameters (Mission Planner UDP)
STREAM_HOST = "192.168.1.155"
STREAM_PORT = 1110
STREAM_FPS = 30
STREAM_BITRATE_KBPS = 1500
RESOLUTION_WIDTH = 1280
RESOLUTION_HEIGHT = 720

last_detections = []
labels = []

class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category
and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata,
picam2)
        self.center_x = None
        self.center_y = None
        self.calculate_center()

    def calculate_center(self):
        """Calculate the center of the bounding box."""
        x, y, w, h = self.box
        self.center_x = x + w / 2
        self.center_y = y + h / 2

class DroneController:
    """ Handles MAVLink communicaiton with the drone """
    
    def __init__(self, drone_ip, drone_port):
        self.drone_ip = drone_ip
        self.drone_port = drone_port
        self.master = None
        self.connected = False

    def connect(self):
        """Connect to the drone via MAVLink"""
        connection_string = f"tcp:{self.drone_ip}:{self.drone_port}"
        print(f"Connecting to drone at {connection_string}...")
        try:
            self.master = mavutil.mavlink_connection(connection_string)
            mavutil.mavlink_connection(connection_string)
            self.master.wait_heartbeat(timeout=10)
            self.connected = True
            print(f"✓ Connected to drone (System ID: {self.master.target_system})")
            return True
        except Exception as e:
            print(f"ERROR: Could not connect to drone: {e}")
            return False
    
    def turn(self, angle: int):
        self.master.command_long_send(self.master.target_system, self.master.target_component, mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0, 45, 25, 0, 0, 0, 0, 0)
        
    def close(self):
        """Close Connection to drone"""
        if self.master:
            try:
                self.master.close()
                print("Drone connection closed")
            except:
                pass


def parse_detections(metadata: dict):
    """Parse the output tensor into detected objects."""
    global last_detections

    # Get outputs from IMX500
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        return last_detections

    # Parse detection results
    boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]

    # Normalize boxes if needed
    input_w, input_h = imx500.get_input_size()
    if intrinsics.bbox_normalization:
        boxes = boxes / input_h

    # Reorder bbox coordinates if needed
    if intrinsics.bbox_order == "xy":
        boxes = boxes[:, [1, 0, 3, 2]]

    # Split boxes into individual coordinates
    boxes = np.array_split(boxes, 4, axis=1)
    boxes = zip(*boxes)

    # Filter detections by threshold
    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > THRESHOLD
    ]
    
    return last_detections

def get_labels():
    """Get label names from intrinsics."""
    labels = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels

def draw_detections(request, stream="main"):
    """Draw detection boxes and labels onto frames before encoding."""
    if not last_detections:
        return

    with MappedArray(request, stream) as m:
        for detection in last_detections:
            x, y, w, h = detection.box
            x, y, w, h = int(x), int(y), int(w), int(h)

            label = ""
            if labels and int(detection.category) < len(labels):
                label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"

            if label:
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                text_x = x + 5
                text_y = y + 15
                overlay = m.array.copy()
                cv2.rectangle(
                    overlay,
                    (text_x, text_y - text_height),
                    (text_x + text_width, text_y + baseline),
                    (255, 255, 255),
                    cv2.FILLED
                )
                cv2.addWeighted(overlay, 0.30, m.array, 0.70, 0, m.array)
                cv2.putText(
                    m.array, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
                )

            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0, 0), 2)


# Section of code for rotating the drone

def calculate_rotation_angle(detection, image_width, image_height):
    """Calculate the rotation angle needed to face the detection center.
    Returns angle in degrees, where 0 is facing forward, positive is clockwise, negative is counterclockwise.
    """
    image_center_x = image_width / 2

    # Calculate horizontal deviation from center
    deviation = detection.center_x - image_center_x

    print("Deviation is: ", deviation)

    # Convert pixel deviation to angle
    degreesPerPixel = HORIZONTAL_FOV_DEG / image_width
    angle = deviation * degreesPerPixel  # Assuming a 60 degree horizontal FOV for the camera

    return angle

# TODO: This is where we'd put in pymavlink commands to actually rotate the drone. For now, it just prints the angle.
def rotate_drone(drone, angle):
    """Rotate the drone by the specified angle (in degrees).
    Angle in degrees (-180 to 180).
    This is a generic example - replace with your drone SDK.
    """
    try:
        # Example for DJI SDK: rotate the drone
        # drone.set_yaw_rate(angle)  # Uncomment with your SDK

        # Or smooth rotation command:
        # drone.rotate(angle, duration=1.0)

        print(f"Rotating drone by {angle:.1f} degrees", flush=True)
    except Exception as e:
        print(f"Error rotating drone: {e}", flush=True)

def track_object(detections, image_width, image_height, drone=None):
    """Track the most confident detection and rotate drone to face it."""
    if not detections:
        return None

    # Get the detection with highest confidence
    best_detection = max(detections, key=lambda d: d.conf)

    # Calculate rotation angle
    angle = calculate_rotation_angle(best_detection, image_width, image_height)

    # Rotate drone
    if drone:
        rotate_drone(drone, angle)
    
    return {
        "target_center": {
            "x": best_detection.center_x,
            "y": best_detection.center_y
        },
        "rotation_angle": angle,
        "confidence": best_detection.conf
    }

def main():
    global picam2, imx500, intrinsics
    
    # Load the object detection model (MobileNet SSD)
    model_path ="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"

    # Initialize IMX500 with the model
    imx500 = IMX500(model_path)
    intrinsics = imx500.network_intrinsics

    # Get image dimensions
    image_width, image_height = RESOLUTION_WIDTH, RESOLUTION_HEIGHT
    print("Image Width is: ", image_width)

    # Load COCO labels
    try:
        with open("/usr/share/imx500-models/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    except:
        # Fallback labels if file not found
        intrinsics.labels = [str(i) for i in range(80)]
    
    intrinsics.update_with_defaults()

    # Get labels
    global labels
    labels = get_labels()

    # Initialize camera
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_video_configuration(
        main={'size': (RESOLUTION_WIDTH, RESOLUTION_HEIGHT)},
        controls={"FrameRate": STREAM_FPS},
        buffer_count=12
    )

    encoder = H264Encoder(bitrate=STREAM_BITRATE_KBPS * 1000)
    output = FfmpegOutput(
        f"-f rtp -payload_type 96 rtp://{STREAM_HOST}:{STREAM_PORT}?pkt_size=1200"
    )

    # Show firmware loading progress
    imx500.show_network_fw_progress_bar()

    # Configure and start camera
    picam2.configure(config)
    picam2.start()
    picam2.pre_callback = draw_detections
    picam2.start_encoder(encoder, output)

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()
    print("IMX500 Object Detection Running - Press Ctrl+C to stop", flush=True)
    print("-" * 60, flush=True)

    try:
        while True:
            # Capture metadata
            metadata = picam2.capture_metadata()

            # Parse detections from metadata
            detections=parse_detections(metadata)

            # Track object and rotate drone
            if detections:
                tracking_info = track_object(detections, image_width, image_height, drone=None)
            
            # Convert to JSON format
            current_detections = []
            for detection in detections:
                x, y, w, h = detection.box
                det_dict = {
                    "class_id": int(detection.category),
                    "class_name": labels[int(detection.category)] if int(detection.category) < len(labels) else
                    f"class_{int(detection.category)}",
                    "confidence": float(detection.conf),
                    "bbox": {
                        "x": float(x),
                        "y": float(y),
                        "width": float(w),
                        "height": float(h)
                    }
                }
                current_detections.append(det_dict)
            
            # Create JSON output
            output = {
                "timestamp": time.time(),
                "detections": current_detections,
                "count": len(current_detections),
                "rotation_from_center_degrees":(
                    tracking_info["rotation_angle"] if tracking_info and detection else None
                ),
                "detections": current_detections
            }

            # Print JSON to terminal
            print(json.dumps(output, indent=2), flush=True)

            # Small delay to avoid overwhelming output
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping object detection...", flush=True)
    finally:
        picam2.stop_encoder()
        picam2.stop()
        print("Camera stopped.", flush=True)

if __name__ == "__main__":
    main()
