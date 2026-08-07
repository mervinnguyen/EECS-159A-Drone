
#Current overlay state
currentMode = "FEED ONLY"
prevTime = time.time()
currentFPS = 0.0

#Update Mode in main function (modify the existing match block)
    match option:
        case "1":
            global currentMode
            currentMode = "FEED + ACTIVE YAW"
            while True:
                enableTracking(flight_controller, picam2, image_width, turn= True)
            
        case "2":
            currentMode = "FEED ONLY"
            while True:
                enableTracking(flight_controller, picam2, image_width, turn = False)
        
        case "q":
            flight_controller.close()
            picam2.stop_encoder()
            picam2.stop()
            print("Camera stopped.", flush=True)
            return

#FPS Calculation (modufy the existing enable_tracking function)
def enableTracking(flight_controller, picam2, image_width, turn):
    global prevTime, currentFPS

    #Get current FPS
    now = time.time()
    currentFPS = 1.0 / (now - prevTime)
    prevTime = now

    #keep existin detection + trackin logic
    metadata = picam2.capture_metadata()
    detections = parse_detections(metadata)

    tracking_info = {"rotation_angle": 0}

    if detections: 
        tracking_info = track_object(detections, image_width) or tracking_info
    
    if turn:
        flight_controller.turn(tracking_info["rotation_angle"])

    time.sleep(3)

def drawDetections(request, stream = "main"):
    global currentMode
    global currentFPS

    with MappedArray(request, stream) as m:
    #Draw Mode + FPS overlap
    cv2.putText(
        m.array,
        f"MODE: {currentMode}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    #FPS Text
    cv2.putText(
        m.array,
        f"FPS: {currentFPS: .2f}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    #keep existing bounding box drawin below
    for detection in last_detections:
        x, y, w, h = detection.box
        x, y, w, h = int(x), int(y), int(w), int(h)

        cv2.rectangle(m.array, (x,y), (x+w, y+h), (0,255,0), 2)

#main
