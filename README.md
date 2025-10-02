# EECS-159A-Drone
Welcome to the EECS 159A: Senior Design Project 1 (Drone) repository. This project guides students through the theory and hands-on practice of building, configuring, and flying a fiber optic drone from start to finish.

At the end of the course, our team will have successfully assembled and flown our own quadcopter, fully equipped with autonomous flight capabilities via ArduPilot.

## 📌 Project Overview

**Course Objective:** Teach fellow students the fundamentals of drone systems (avionics, power, propulsion, control, telemetry, and safety) through theory paired with labs.

**Hands-On Learning:** Each week introduces new concepts, followed by integration of software/firmware and testing of a new drone component.

**Flight Demonstration:** Final evaluation requires students to showcase a fully autonomous GPS waypoint mission with telemetry feedback to a ground control station.

## 🔧 Fiber Optic Drone System Breakdown

The drone is a multirotor quadcopter built using standardized department-supplied kits. The build process is broken into major subsystems:

**1. Airframe**

- Provides structure and mounting points for all components.

- Typically a lightweight carbon fiber or composite X-frame.

**2. Motors & Propellers**

- Brushless DC Motors (BLDC): Provide thrust.

- Propellers: Sized and pitched for balance between efficiency and stability.

**3. Electronic Speed Controllers (ESCs)**

- Regulate motor power based on signals from the flight controller.

- Handle rapid throttle adjustments for stability.

**4. Power System**

- LiPo Battery: High-discharge power source.

- Power Distribution Board (PDB): Safely routes battery power to ESCs and other electronics.

- Battery chargers & safety protocols emphasized.

**5. Flight Controller (FC)**

- The “brain” of the drone.

- Runs ArduPilot firmware for stabilization, flight modes, GPS navigation, and waypoint missions.

- Integrates IMU (gyroscope + accelerometer), compass, barometer, and GPS.

**6. Telemetry & Ground Control Station (GCS)**

- Telemetry radios send real-time flight data to the GCS.

- GCS software (e.g., Mission Planner or QGroundControl) is used for mission planning, waypoint upload, and monitoring.

**7. IR Camera**

- Infrared camera module for thermal imaging and low-light vision.

- Enables applications such as obstacle detection, search-and-rescue, and environmental monitoring.

- Data can be processed onboard (via Raspberry Pi) or streamed to the ground station.

**8. AI Camera**

- Equipped with machine learning capabilities for object detection and recognition.

- Supports autonomous decision-making in tasks such as target tracking and path planning.

- Can integrate with ROS or other AI frameworks for advanced autonomy.

**9. Raspberry Pi 4 (Companion Computer)**

- Acts as a high-performance companion computer to the flight controller.

- Enables advanced onboard processing, including computer vision and AI algorithms.

- Provides additional connectivity for sensors, cameras, and custom applications. 
  
**10. Safety Systems**

- Geofencing, return-to-launch (RTL), and emergency failsafes.

- Preflight checklist required before each flight.
