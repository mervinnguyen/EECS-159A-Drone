# Object_Tracking
The Object_Tracking.py script does the following:

1. Utilizes a built in object detection model with the Raspberry Pi AI Camera (IMX500 sensor)
2. Detects objects using the model. It will classify the object and return a bounding box position
3. Stream live video to Mission Planner.
4. Calculate and return the Angle the drone should rotate

![alt text](assets/image.png)
Example output and video feed on Mission Planner

TODO:

- [ ] Complete Pymavlink integration

- [ ] Fix returned angle. Needs to factor in camera FOV

## Setup

To setup the script we will first need to setup a Python virtual environment as such:

**1. Create a virtual environment in "Vision Tracking pipeline" directory:**

` python3 -m venv .venv --system-site-packages `

**2. Source and Activate the Virtual Enviornment:**

`source .venv/bin/activate`

**3. Download Required Python Libraries:**

`pip install -r requirements.txt`

*We are assuming that picamera2 and all of the raspberry pi ai camera dependencies have been installed system-wide and are under site-packages*

## Running The Script

To run the script simply run this command under the "Vision Tracking pipeline" directory:

`python3 Object_Tracking.py`

You should start to see that the firmware is being flashed to the camera sensor and JSON objects being returned about what objects are detected.

### Connecting to Mission Planner

The process to connect the object detection feed to Mission Planner is very similar to how we've previously done it. *This is assuming the ground station has a static IP of 192.168.1.155*

**1. Run The Script**

**2. Copy GStreamer String:**

`udpsrc port=1110 caps="application/x-rtp, media=video, encoding-name=H264, payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=2 ! videoconvert ! video/x-raw,format=BGRA ! appsink name=outsink sync=false`

And that's it, the feed should be showing on Mission Planner!