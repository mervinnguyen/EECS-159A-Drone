# PhotonFlight: Fiber-Optic Autonomous UAV

PhotonFlight is a **fiber-optic-tethered autonomous quadcopter** designed for operation in **RF-denied or GPS-degraded environments**.  
The platform replaces traditional RF telemetry with a **secure bidirectional SFP fiber link**, enabling **low-latency video streaming, real-time telemetry, and reliable command/control communication** between the UAV and ground control station.

The system integrates **ArduPilot flight control, embedded Linux module, onboard AI vision processing, and fiber-optic networking**, demonstrating a complete embedded system spanning hardware, firmware, networking, and autonomy.

---

# Abstract

PhotonFlight is a **fiber-optic tethered UAV platform** developed for search-and-rescue and infrastructure inspection applications where traditional RF communication may be unreliable or unavailable.

The system leverages **single-mode fiber optic communication via SFP transceivers** to provide:

- RF-interference immunity  
- secure high-bandwidth communication  
- low-latency telemetry and video streaming  

An onboard AI camera enables **real-time object detection and autonomous tracking**, while a companion computer processes video and coordinates communication with the flight controller.

The result is a **robust embedded UAV platform capable of operating in communication-contested environments**.

---

# System Architecture

PhotonFlight is composed of several integrated system layers that enable flight control, telemetry transmission, and onboard intelligence.


Each layer performs a specific role in the overall system.

| Layer | Responsibility |
|------|------|
| Ground Control Station | Mission planning, telemetry monitoring, system control |
| Fiber Communication Layer | Secure telemetry and high-bandwidth video transmission |
| Companion Computer | AI inference, video processing, communication bridging |
| Flight Controller | Real-time stabilization, sensor fusion, flight control |
| Motor Control | Direct actuation of propulsion system |

<img width="6109" height="5374" alt="e" src="https://github.com/user-attachments/assets/7ba3364f-91c0-4ff5-98d2-8df0399958b6" />
---

# Hardware Architecture

PhotonFlight is built using a combination of **commercial off-the-shelf components and custom fabricated hardware**.

## Airframe

The UAV uses an **8-inch quadcopter carbon fiber frame** providing structural mounting for propulsion, avionics, and communication hardware.

Key design considerations include:

- vibration isolation
- weight optimization
- secure component mounting

---

## Propulsion System

The propulsion system consists of:

- **Brushless DC motors (BLDC)**
- **high-efficiency propellers**
- **electronic speed controllers (ESCs)**

ESCs receive control signals from the flight controller to regulate thrust and maintain stability.

---

## Power System

The drone is powered by a **high-discharge LiPo battery** distributed through a **Power Distribution Board (PDB)**.

The power subsystem manages:

- high-current delivery
- electrical isolation
- voltage regulation for onboard electronics

Battery safety procedures and charge management protocols are implemented for safe operation.

---

# Flight Control System

The UAV uses an **ArduPilot-based flight controller** responsible for deterministic flight control.

Responsibilities include:

- attitude stabilization
- sensor fusion
- waypoint navigation
- flight mode control
- failsafe management

The controller integrates multiple onboard sensors:

- IMU (accelerometer + gyroscope)
- magnetometer
- barometer
- GPS receiver

Flight control commands are communicated via **MAVLink protocol**.

---

# Companion Computer

A **Raspberry Pi 4** acts as the onboard companion computer, providing high-level processing capabilities.

The companion computer is responsible for:

- AI inference and object detection
- video encoding and streaming
- communication routing between sensors and ground station
- system coordination

The Raspberry Pi communicates with the flight controller through **MAVLink interfaces**.

---

# Fiber-Optic Communication Pipeline

PhotonFlight replaces traditional RF telemetry with a **fiber-optic communication architecture**.

The system uses:

- **SFP transceivers**
- **single-mode fiber optic cable**
- **media converters**

This communication architecture provides several advantages:

- immunity to RF interference
- secure communication channel
- high-bandwidth data transmission
- reduced telemetry latency

Data transmitted through the fiber link includes:

- MAVLink telemetry
- AI camera video feed
- mission control commands

---

# Vision System

PhotonFlight incorporates two vision modules enabling perception capabilities.

## Infrared Camera

The infrared camera enables **thermal imaging and low-light operation**, allowing the UAV to function in environments such as:

- disaster response zones
- low visibility environments
- nighttime search-and-rescue missions

---

## AI Camera

The onboard AI camera performs **real-time object detection and tracking** using pretrained neural network models.

Capabilities include:

- autonomous target tracking
- real-time object detection
- visual feedback to ground operators

Inference can be executed onboard or streamed for external processing.

---

# Ground Control Station

A ground control station running **Mission Planner or QGroundControl** enables operators to:

- monitor real-time telemetry
- upload mission waypoints
- visualize system status
- issue control commands

Communication between the UAV and the ground station occurs over the fiber link.

---

# Safety Systems

Several redundant safety mechanisms are implemented to ensure safe operation.

These include:

- geofencing
- return-to-launch (RTL)
- emergency failsafe procedures
- pre-flight validation checklists

These systems mitigate risk during autonomous flight operations.

---

# Key Technologies

PhotonFlight integrates multiple hardware and software technologies:

**Flight Control**
- ArduPilot
- MAVLink

**Embedded Systems**
- Raspberry Pi 4
- embedded Linux

**Networking**
- fiber-optic SFP communication
- telemetry streaming

**Perception**
- onboard AI camera
- computer vision inference

---

# Documentation & Project Resources

Project documentation and development artifacts are maintained collaboratively.

Resources include:

- system architecture documentation
- hardware integration notes
- flight test logs
- telemetry data analysis
- design review presentations
- safety procedures and compliance documentation

---

# Project Focus

PhotonFlight demonstrates the integration of:

- embedded flight control systems  
- secure communication infrastructure  
- onboard AI perception  

into a UAV platform capable of operating in **RF-contested or GPS-degraded environments**.

This project highlights the application of **embedded systems engineering, networking architecture, and autonomous robotics** in real-world mission scenarios.


