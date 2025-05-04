
#!/usr/bin/env python3
"""
Sentinel Eye Mobile Guard - Raspberry Pi Controller
A security robot implementation with camera, sensors and autonomous movement
"""

import time
import threading
import json
import os
import signal
import sys
from datetime import datetime
import cv2
import numpy as np
import RPi.GPIO as GPIO
from gpiozero import DistanceSensor, MotionSensor
import socketio
from flask import Flask, Response, request

# ----- Configuration -----
CONFIG = {
    "motor": {
        "left_forward": 17,
        "left_backward": 18,
        "right_forward": 22,
        "right_backward": 23,
        "pwm_left": 19,
        "pwm_right": 26,
    },
    "sensors": {
        "pir_pin": 25,
        "ultrasonic_echo": 24,
        "ultrasonic_trigger": 27,
        "infrared_sensor": 16,
    },
    "camera": {
        "resolution": (640, 480),
        "framerate": 20,
        "rotation": 180,  # Set to 0, 90, 180, or 270 as needed
    },
    "patrol": {
        "default_speed": 70,  # PWM value (0-100)
        "scanning_speed": 40,
    },
    "server": {
        "host": "0.0.0.0",
        "port": 5000,
    },
    "detection": {
        "confidence_threshold": 0.5,
        "save_detections": True,
        "detection_folder": "detections",
    }
}

# ----- Initialize GPIO -----
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Motor pins setup
for pin in CONFIG["motor"].values():
    GPIO.setup(pin, GPIO.OUT)

# Initialize PWM for motor speed control
pwm_left = GPIO.PWM(CONFIG["motor"]["pwm_left"], 100)
pwm_right = GPIO.PWM(CONFIG["motor"]["pwm_right"], 100)
pwm_left.start(0)
pwm_right.start(0)

# Initialize sensors
pir_sensor = MotionSensor(CONFIG["sensors"]["pir_pin"])
ultrasonic_sensor = DistanceSensor(
    echo=CONFIG["sensors"]["ultrasonic_echo"],
    trigger=CONFIG["sensors"]["ultrasonic_trigger"]
)
GPIO.setup(CONFIG["sensors"]["infrared_sensor"], GPIO.IN)

# Initialize camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera"]["resolution"][0])
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera"]["resolution"][1])
camera.set(cv2.CAP_PROP_FPS, CONFIG["camera"]["framerate"])

# ----- Global State -----
class RobotState:
    def __init__(self):
        self.status = "online"  # online, patrolling, charging, alert, offline, maintenance
        self.patrol_mode = "auto"  # auto, manual, scheduled, perimeter, follow
        self.battery_level = 100
        self.last_detection = None
        self.alerts = []
        self.current_location = "Home Base"
        self.motion_detected = False
        self.obstacle_distance = float('inf')
        self.patrol_active = False
        self.manual_control = False
        self.zones = [
            {"id": "zone1", "name": "Front Door", "priority": 1},
            {"id": "zone2", "name": "Backyard", "priority": 2},
            {"id": "zone3", "name": "Living Room", "priority": 3}
        ]
        self.current_zone_id = "zone1"

# Initialize robot state
robot = RobotState()

# ----- Motor Control Functions -----
def stop_motors():
    """Stop all motors"""
    GPIO.output(CONFIG["motor"]["left_forward"], GPIO.LOW)
    GPIO.output(CONFIG["motor"]["left_backward"], GPIO.LOW)
    GPIO.output(CONFIG["motor"]["right_forward"], GPIO.LOW)
    GPIO.output(CONFIG["motor"]["right_backward"], GPIO.LOW)
    pwm_left.ChangeDutyCycle(0)
    pwm_right.ChangeDutyCycle(0)

def move_forward(speed=None):
    """Move the robot forward"""
    if speed is None:
        speed = CONFIG["patrol"]["default_speed"]
    
    stop_motors()
    GPIO.output(CONFIG["motor"]["left_forward"], GPIO.HIGH)
    GPIO.output(CONFIG["motor"]["right_forward"], GPIO.HIGH)
    pwm_left.ChangeDutyCycle(speed)
    pwm_right.ChangeDutyCycle(speed)

def move_backward(speed=None):
    """Move the robot backward"""
    if speed is None:
        speed = CONFIG["patrol"]["default_speed"]
    
    stop_motors()
    GPIO.output(CONFIG["motor"]["left_backward"], GPIO.HIGH)
    GPIO.output(CONFIG["motor"]["right_backward"], GPIO.HIGH)
    pwm_left.ChangeDutyCycle(speed)
    pwm_right.ChangeDutyCycle(speed)

def turn_left(speed=None):
    """Turn the robot left"""
    if speed is None:
        speed = CONFIG["patrol"]["default_speed"]
    
    stop_motors()
    GPIO.output(CONFIG["motor"]["right_forward"], GPIO.HIGH)
    pwm_right.ChangeDutyCycle(speed)

def turn_right(speed=None):
    """Turn the robot right"""
    if speed is None:
        speed = CONFIG["patrol"]["default_speed"]
    
    stop_motors()
    GPIO.output(CONFIG["motor"]["left_forward"], GPIO.HIGH)
    pwm_left.ChangeDutyCycle(speed)

def rotate(direction, speed=None):
    """Rotate the robot in place"""
    if speed is None:
        speed = CONFIG["patrol"]["scanning_speed"]
    
    stop_motors()
    if direction == "left":
        GPIO.output(CONFIG["motor"]["left_backward"], GPIO.HIGH)
        GPIO.output(CONFIG["motor"]["right_forward"], GPIO.HIGH)
    else:  # right
        GPIO.output(CONFIG["motor"]["left_forward"], GPIO.HIGH)
        GPIO.output(CONFIG["motor"]["right_backward"], GPIO.HIGH)
    
    pwm_left.ChangeDutyCycle(speed)
    pwm_right.ChangeDutyCycle(speed)

# ----- Object Detection with OpenCV -----
# Load pre-trained model for object detection (using YOLO)
def load_yolo():
    """Load YOLO model for object detection"""
    try:
        # Paths to YOLO configuration and weights files
        # These would be stored in your Raspberry Pi
        yolo_weights = "yolov4-tiny.weights"
        yolo_cfg = "yolov4-tiny.cfg"
        
        # Load YOLO network
        net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
        
        # Get the output layer names
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        
        # Load COCO class labels
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        
        return net, output_layers, classes
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None, None, None

# Object detection function
def detect_objects(frame, net, output_layers, classes):
    """Detect objects in the frame using YOLO"""
    if net is None:
        return frame, []
    
    height, width, _ = frame.shape
    
    # Detect objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    # Information to return
    class_ids = []
    confidences = []
    boxes = []
    detected_objects = []
    
    # Process detection results
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > CONFIG["detection"]["confidence_threshold"]:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maxima suppression to remove redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Draw bounding boxes and add detected objects to the list
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            
            # Define color based on object type
            if label in ['person']:
                color = (0, 255, 0)  # Green for humans
                detection_type = "human"
            elif label in ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear']:
                color = (255, 165, 0)  # Orange for animals
                detection_type = "animal"
            elif label in ['car', 'truck', 'bicycle', 'motorcycle', 'bus']:
                color = (0, 0, 255)  # Red for vehicles
                detection_type = "vehicle"
            else:
                color = (255, 255, 0)  # Yellow for other objects
                detection_type = "object"
                
            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add to detected objects list
            object_info = {
                "id": f"obj_{int(time.time())}_{i}",
                "type": detection_type,
                "confidence": confidence,
                "boundingBox": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h
                },
                "tracked": False,
                "label": label
            }
            detected_objects.append(object_info)
            
            # Create an alert for person detection with high confidence
            if detection_type == "human" and confidence > 0.7:
                create_alert(detection_type, robot.current_zone_id)
    
    return frame, detected_objects

# Save detection image
def save_detection(frame, objects):
    """Save frame with detected objects"""
    if CONFIG["detection"]["save_detections"] and objects:
        os.makedirs(CONFIG["detection"]["detection_folder"], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{CONFIG['detection']['detection_folder']}/detection_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        return filename
    return None

# ----- Alert System -----
def create_alert(detection_type, location):
    """Create a new alert"""
    alert_id = f"alert_{int(time.time())}"
    timestamp = datetime.now()
    
    # Determine alert level based on detection type
    if detection_type == "human":
        level = "critical"
    elif detection_type in ["animal", "vehicle"]:
        level = "warning"
    else:
        level = "info"
    
    # Get the zone name from the zone ID
    zone_name = next((zone["name"] for zone in robot.zones if zone["id"] == location), "Unknown")
    
    alert = {
        "id": alert_id,
        "timestamp": timestamp.isoformat(),
        "type": detection_type,
        "location": zone_name,
        "level": level,
        "acknowledged": False
    }
    
    robot.alerts.append(alert)
    robot.status = "alert"
    
    # If we have more than 100 alerts, remove the oldest ones
    if len(robot.alerts) > 100:
        robot.alerts = robot.alerts[-100:]
    
    print(f"Alert created: {detection_type} detected in {zone_name}")
    return alert_id

# ----- Sensor Monitoring -----
def read_sensors():
    """Read all sensors and update robot state"""
    # Read PIR motion sensor
    motion_detected = pir_sensor.motion_detected
    
    # Read ultrasonic distance sensor
    distance = ultrasonic_sensor.distance * 100  # Convert to cm
    
    # Read infrared sensor
    infrared_detected = not GPIO.input(CONFIG["sensors"]["infrared_sensor"])
    
    # Update robot state
    robot.motion_detected = motion_detected or infrared_detected
    robot.obstacle_distance = distance
    
    # Take action based on sensors
    if robot.patrol_active:
        # If obstacle detected, avoid it
        if distance < 30:  # Less than 30cm
            print(f"Obstacle detected at {distance:.1f}cm")
            # Simple obstacle avoidance
            stop_motors()
            move_backward()
            time.sleep(1)
            rotate("right")
            time.sleep(1.5)
            stop_motors()
        
        # If motion is detected, pause and scan
        elif robot.motion_detected and not robot.manual_control:
            print("Motion detected, scanning...")
            stop_motors()
            robot.status = "alert"
            # Rotate to scan the area
            rotate("left", 30)
            time.sleep(3)
            stop_motors()
            robot.status = "patrolling"

# ----- Battery Simulation -----
def simulate_battery():
    """Simulate battery drain"""
    while True:
        # Decrease battery level when patrolling
        if robot.status in ["patrolling", "alert"]:
            robot.battery_level = max(0, robot.battery_level - 0.1)
        elif robot.status == "charging":
            robot.battery_level = min(100, robot.battery_level + 0.5)
        
        # Set to charging if battery is low
        if robot.battery_level < 15 and robot.status != "charging":
            print("Battery low, returning to charging station")
            robot.status = "charging"
            robot.patrol_active = False
            # Implement return to charging station logic here
        
        # Fully charged
        if robot.battery_level >= 100 and robot.status == "charging":
            robot.status = "online"
        
        time.sleep(5)

# ----- Patrol Logic -----
def patrol_thread():
    """Autonomous patrol logic"""
    while True:
        if robot.patrol_active and not robot.manual_control:
            # Only patrol if not manually controlled and battery is not low
            if robot.status == "patrolling":
                # Simple patrol pattern
                move_forward()
                time.sleep(3)
                stop_motors()
                rotate("right")
                time.sleep(1)
                stop_motors()
                time.sleep(1)
            
            # Regularly update sensor readings
            read_sensors()
        
        time.sleep(0.5)

# ----- Flask Web Server -----
app = Flask(__name__)
sio = socketio.Server(cors_allowed_origins='*')
app = socketio.WSGIApp(sio, app)

# Socket.IO events
@sio.event
def connect(sid, environ):
    print(f"Client connected: {sid}")
    sio.emit('status_update', get_status(), to=sid)

@sio.event
def disconnect(sid):
    print(f"Client disconnected: {sid}")

@sio.event
def control_command(sid, data):
    """Handle control commands from client"""
    cmd_type = data.get('type')
    params = data.get('params', {})
    
    print(f"Received command: {cmd_type}, params: {params}")
    
    if cmd_type == 'move':
        direction = params.get('direction')
        speed = params.get('speed', CONFIG["patrol"]["default_speed"])
        robot.manual_control = True
        
        if direction == 'forward':
            move_forward(speed)
        elif direction == 'backward':
            move_backward(speed)
        elif direction == 'left':
            turn_left(speed)
        elif direction == 'right':
            turn_right(speed)
    
    elif cmd_type == 'stop':
        stop_motors()
        robot.manual_control = False
    
    elif cmd_type == 'patrol':
        zone_id = params.get('zone')
        robot.status = 'patrolling'
        robot.patrol_active = True
        robot.manual_control = False
        if zone_id:
            robot.current_zone_id = zone_id
    
    elif cmd_type == 'return':
        robot.status = 'returning'
        # Logic to return to home base
        stop_motors()
        robot.patrol_active = False
        robot.manual_control = False
    
    sio.emit('status_update', get_status())
    return {'success': True}

@sio.event
def acknowledge_alert(sid, data):
    """Mark an alert as acknowledged"""
    alert_id = data.get('id')
    if alert_id:
        for alert in robot.alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                break
    
    return {'success': True}

# API routes
@app.route('/status')
def get_status():
    """Get current robot status"""
    return {
        'status': robot.status,
        'patrolMode': robot.patrol_mode,
        'batteryLevel': robot.battery_level,
        'currentLocation': robot.current_location,
        'motionDetected': robot.motion_detected,
        'obstacleDistance': robot.obstacle_distance,
        'patrolActive': robot.patrol_active,
        'manualControl': robot.manual_control,
        'alerts': robot.alerts
    }

@app.route('/alerts')
def get_alerts():
    """Get all alerts"""
    return {'alerts': robot.alerts}

@app.route('/zones')
def get_zones():
    """Get patrol zones"""
    return {'zones': robot.zones}

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    # This will be a streaming response for video
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Video streaming generator function
def generate_frames():
    """Generate frames from camera with object detection"""
    # Load YOLO model
    net, output_layers, classes = load_yolo()
    
    while True:
        success, frame = camera.read()
        if not success:
            break
            
        # Apply rotation if needed
        if CONFIG["camera"]["rotation"] != 0:
            if CONFIG["camera"]["rotation"] == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif CONFIG["camera"]["rotation"] == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif CONFIG["camera"]["rotation"] == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
        # Object detection
        processed_frame, detected_objects = detect_objects(frame, net, output_layers, classes)
        
        # If objects detected, save the image and update robot state
        if detected_objects:
            image_path = save_detection(processed_frame, detected_objects)
            robot.last_detection = {
                'timestamp': datetime.now().isoformat(),
                'objects': detected_objects,
                'image_path': image_path
            }
            # Emit detection event via Socket.IO
            sio.emit('object_detected', {
                'timestamp': robot.last_detection['timestamp'],
                'objects': detected_objects
            })
            
        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        # Yield frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
# ----- Main Program -----
def main():
    # Create detection folder if it doesn't exist
    if CONFIG["detection"]["save_detections"]:
        os.makedirs(CONFIG["detection"]["detection_folder"], exist_ok=True)
    
    # Start background threads
    threading.Thread(target=simulate_battery, daemon=True).start()
    threading.Thread(target=patrol_thread, daemon=True).start()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("Shutting down...")
        stop_motors()
        camera.release()
        GPIO.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the server
    print(f"Starting server on {CONFIG['server']['host']}:{CONFIG['server']['port']}")
    from waitress import serve
    serve(app, host=CONFIG["server"]["host"], port=CONFIG["server"]["port"])

if __name__ == "__main__":
    main()
