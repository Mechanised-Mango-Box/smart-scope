import os
import cv2
import numpy as np
import io
import logging
import socketserver
from http import server
from threading import Condition, Thread

import scipy

# Constants
HUD_GREEN = (0, 255, 0)
HUD_THICKNESS = 2
HUD_FONT_SIZE = 1  # 0.4

# screen is 4:3
VID_IN_HEIGHT = 800
VID_IN_WIDTH = 600
VID_OUT_HEIGHT = 320
VID_OUT_WIDTH = 240

ACC_THRESHOLD = 0.55  # Threshold to detect objects

# Paths
PATH = os.getcwd()
classFile = os.path.join(PATH, "coco.names")
configPath = os.path.join(PATH, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
weightsPath = os.path.join(PATH, "frozen_inference_graph.pb")

# Load class names
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Initialize the network
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(VID_IN_WIDTH, VID_IN_HEIGHT)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


# Function to detect objects
def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    if len(objects) == 0:
        objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=HUD_GREEN, thickness=HUD_THICKNESS)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, HUD_FONT_SIZE, HUD_GREEN, HUD_THICKNESS)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, HUD_FONT_SIZE, HUD_GREEN, HUD_THICKNESS)
    return img, objectInfo


if __name__ == "__main__":
    # Initialize the video capture
    cap = cv2.VideoCapture("http://192.168.0.127:8000/stream.mjpg")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VID_IN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VID_IN_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 10)

    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            result, objectInfo = getObjects(img, ACC_THRESHOLD, 0.2)

            # Encode frame as JPEG
            # stream = np.flipud(result)

            cv2.imshow("Output - CamFeed", result)
            # cv2.imshow("Output - Obj ID", stream)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()