
# Sentinel Eye Mobile Guard - Raspberry Pi Implementation

This repository contains the Python code for running a security robot using a Raspberry Pi, motors, camera, and various sensors.

## Hardware Requirements

- Raspberry Pi 4 (recommended) or 3B+
- Robot car chassis with motors (2 or 4 wheel drive)
- Camera module (Raspberry Pi Camera or USB webcam)
- Sensors:
  - PIR motion sensor
  - Ultrasonic distance sensor (HC-SR04)
  - Infrared sensor
- L298N motor driver (or similar)
- Power supply for Raspberry Pi and motors (battery pack recommended)
- Optional: servo motors for camera pan/tilt

## Software Dependencies

Install the required libraries:

```bash
sudo apt-get update
sudo apt-get install -y python3-opencv python3-gpiozero python3-rpi.gpio
pip3 install flask socketio waitress numpy
```

### Download YOLO model files:

```bash
# Download YOLOv4-tiny weights
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights

# Download YOLOv4-tiny config
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg

# Download COCO names file
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
```

## Wiring Instructions

### Motor Connections:

Connect the L298N motor driver to the Raspberry Pi GPIO pins:
- Left Motor Forward: GPIO 17
- Left Motor Backward: GPIO 18
- Right Motor Forward: GPIO 22
- Right Motor Backward: GPIO 23
- PWM Left: GPIO 19
- PWM Right: GPIO 26

### Sensor Connections:

- PIR Motion Sensor: GPIO 25
- Ultrasonic Sensor Trigger: GPIO 27
- Ultrasonic Sensor Echo: GPIO 24
- Infrared Sensor: GPIO 16

## Running the Application

1. Clone this repository to your Raspberry Pi
2. Install the required dependencies
3. Run the script:

```bash
python3 robot_security_system.py
```

## Web Interface

The robot exposes a web interface on port 5000. You can control the robot and view its status by connecting to:

```
http://[RASPBERRY_PI_IP]:5000
```

## Mobile App Integration

To connect the mobile app to the Raspberry Pi:

1. Make sure the Raspberry Pi and mobile device are on the same network
2. In the mobile app settings, enter the Raspberry Pi's IP address and port
3. The mobile app will connect to the robot system via Socket.IO

## Features

- Real-time video streaming with AI-based object detection
- Multiple motion detection sensors (PIR, ultrasonic, infrared)
- Autonomous patrol mode with obstacle avoidance
- Manual control via mobile app
- Alert system for detected objects
- Battery monitoring
- Cloud storage for detected objects

## Customization

You can modify the configuration parameters in the `CONFIG` dictionary at the top of the script to adjust:
- GPIO pin assignments
- Camera settings
- Patrol behavior
- Detection sensitivity
- Server configuration

## Troubleshooting

- Ensure all GPIO pins are correctly connected
- Check that all sensors are functioning properly
- Make sure the camera is enabled in Raspberry Pi config
- Verify that the YOLO model files are in the same directory as the script

## Safety Notes

- The robot should be supervised during operation
- Always test in a safe environment first
- Be careful with power supply connections to prevent damage to the Raspberry Pi
