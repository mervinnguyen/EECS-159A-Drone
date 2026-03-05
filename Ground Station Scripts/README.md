# Ground Station Scripts
This folder is going to house all of the Python scripts that need to run in a WSL instance if **the Pi on the drone is not able to do object detection**.

This folder should consist of 1 Python script that is able to do the following tasks:

1. Receive a video stream from the Pi
2. Detect objects and categorize objects in each frame
3. Determine what object to focus on 
4. Calculate the center of the object
5. Find out the angle the drone needs to rotate in order to face the center of the detected object
6. Send a Mavlink command to the Flight Controller so that the drone can turn