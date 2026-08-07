# PhotonFlight: Fiber-Optic Autonomous UAV

PhotonFlight is a **fiber-optic-tethered autonomous quadcopter** designed for operation in **RF-denied or GPS-degraded environments**.  

The platform replaces traditional RF telemetry with a **secure bidirectional SFP fiber link**, enabling low-latency video streaming, real-time telemetry, and **reliable command/control communication** between the UAV and ground control station.

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

# Demo

The video below shows the drone through its takeoff, tracking, and landing sequence. The drone first takes off in **Loiter** mode, switches to **Guided** mode, and then back to **Loiter** for landing. The tracking script is active at all times, but the flight controller only accepts the flight commands in **Guided** mode. The entire demo was shown without any wireless/radio communications. All telemetry data, video, and joystick input were sent over a series of wired connections to the drone. The only wireless signal emitted from the drone was GPS for position tracking.

https://github.com/user-attachments/assets/d5699f7d-33b4-4451-9186-9bc827cc7043

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

The UAV uses an **8-inch quadcopter frame** providing structural mounting for propulsion, avionics, and communication hardware. It is made from nylon which allows us to have a relatively light and modular frame while keeping our cost low. This can be improved in the future by using an all carbon fiber frame. 

---

## Propulsion System

The propulsion system consists of:

- **4 Brushless DC motors (BLDC)**
- **High-efficiency propellers**
- **4IN1 electronic speed controllers (ESCs)**

ESCs receive control signals from the flight controller (FC) and translates them into how much power will be delievered to each motor. The FC talks to the ESC over a protocol known as DSHOT, which is a digitial protocol that aims to solve issues with traditional PWM (Pulse Width Modulation) protocols. The main benefit is that it relies on a digital signal over an analog one, which means it is more resistant to noise and oscillator speed variations.

---

## Power System

<img width="7793" height="5375" alt="e(2)" src="https://github.com/user-attachments/assets/473ca380-0b09-4dcc-a940-32ef571e235d" />

The drone is powered by a **6S1P LiPo battery** (6 cells in series) distributed through a PCB integrated on the frame. On top of this we have step down converters to power our raspberry Pi and our SFP to Ethernet converter.

Voltage Ranges for our Devices:
- 22.2V: ESC and Flight Controller (has its own voltage regulator)
- 12V: SFP to Ethernet Converter
- 5V: Raspberry Pi and AI Enabled Camera

Battery safety procedures and charge management protocols are implemented for safe operation. We ensure that our batteries are regularly balanced and are charged under supervision. LiPo batteries carry many risks due to its chemistry and energy dense nature. We keep them in fire resistant bags as a precaution.

---

# Flight Control System

The UAV uses an **Matek F405** responsible for deterministic flight control.

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

![e](https://github.com/user-attachments/assets/95bf0324-b551-4cad-8efd-d9007c29ff98)


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


