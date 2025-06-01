#!/usr/bin/env python3

import cv2
import mediapipe as mp
import time
import numpy as np
from djitellopy import Tello
from face_tracker.PlotModule import LivePlot

def PIDController(PID, img, target, cVal, limit=[-100,100], draw=False):
    """
    PID controller that returns clipped int control value.
    """
    # static vars
    if not hasattr(PIDController, "pError"):
        PIDController.pError = 0
        PIDController.pTime = time.time()
        PIDController.I = 0

    t  = time.time() - PIDController.pTime
    err = target - cVal
    P   = PID[0] * err
    PIDController.I += PID[1] * err * t
    D   = PID[2] * (err - PIDController.pError) / (t or 1e-6)
    val = P + PIDController.I + D
    val = float(np.clip(val, limit[0], limit[1]))
    if draw:
        cv2.putText(img, str(int(val)), (50,70),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255,0,255),3)
    PIDController.pError = err
    PIDController.pTime  = time.time()
    return int(val)

def main():
    # parameters
    width, height = 640, 480
    xPID = [0.21, 0, 0.1]
    yPID = [0.27, 0, 0.1]
    zPID = [0.0021, 0, 0.1]
    xTarget, yTarget, zTarget = width//2, height//2, 11500

    # live‐plot
    plotX = LivePlot(yLimit=[-width//2, width//2], char='X')
    plotY = LivePlot(yLimit=[-height//2, height//2], char='Y')
    plotZ = LivePlot(yLimit=[-100,100], char='Z')

    # MediaPipe face detector
    mpFaces = mp.solutions.face_detection
    faces   = mpFaces.FaceDetection(min_detection_confidence=0.5, model_selection=1)

    # Tello setup
    drone = Tello()
    drone.connect()
    print(f"Battery: {drone.get_battery()}%")
    drone.streamoff()
    drone.streamon()
    drone.takeoff()
    time.sleep(2)
    drone.move_up(80)

    try:
        while True:
            frame = drone.get_frame_read().frame
            frame = cv2.resize(frame, (width, height))
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = faces.process(imgRGB)

            if results.detections:
                d = results.detections[0].location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(d.xmin*iw), int(d.ymin*ih), int(d.width*iw), int(d.height*ih)
                cx, cy = x + w//2, y + h//2
                area = w*h

                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.circle(frame, (cx,cy), 5, (255,255,0), cv2.FILLED)

                xVal = PIDController(xPID, frame, xTarget, cx)
                yVal = PIDController(yPID, frame, yTarget, cy)
                zVal = PIDController(zPID, frame, zTarget, area, limit=[-20,15], draw=True)

                imgPX = plotX.update(xVal)
                imgPY = plotY.update(yVal)
                imgPZ = plotZ.update(zVal)

                top = np.hstack((frame, imgPX))
                bot = np.hstack((imgPY, imgPZ))
                display = np.vstack((top, bot))

                drone.send_rc_control(0, zVal, yVal, -xVal)
            else:
                display = np.zeros((height*2, width*2, 3), np.uint8)

            cv2.imshow("Tello Face Tracker", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        drone.land()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
