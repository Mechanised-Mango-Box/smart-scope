import os
import time
import numpy as np
import json
import cv2
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

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


def opencv_rect_to_pil_rect(opencv_rect):
    x, y, width, height = opencv_rect
    left = x
    upper = y
    right = x + width
    lower = y + height
    return [left, upper, right, lower]


class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # only executes upon request

        # Continuous frame processing
        while True:
            success, img = cap.read()
            if not success:
                break

            result, objectInfo = getObjects(img, ACC_THRESHOLD, 0.2)
            cv2.imshow("Output - CamOverlayed", result)

            if len(objectInfo) > 0:
                try:
                    # transform to json to stream
                    print(objectInfo)  # serialise this

                    # Send bounding boxes as JSON
                    for item in objectInfo:  # coords are top left then bottom right
                        item[0] = opencv_rect_to_pil_rect(item[0].tolist())

                        item[0][0] *= VID_OUT_WIDTH / VID_IN_WIDTH
                        item[0][1] *= VID_OUT_HEIGHT / VID_IN_HEIGHT
                        item[0][2] *= VID_OUT_WIDTH / VID_IN_WIDTH
                        item[0][3] *= VID_OUT_HEIGHT / VID_IN_HEIGHT
                        # item[0][0] = 0
                        # item[0][1] = 0
                        # item[0][2] = VID_OUT_WIDTH
                        # item[0][3] = VID_OUT_HEIGHT

                    json_data = json.dumps(objectInfo)  # FIX THIS
                    self.send_response(200)  # HTTP status code 200 OK
                    self.send_header("Content-Type", "application/json")  # Set content type to JSON
                    self.send_header("Content-Length", str(len(json_data)))  # Set content length
                    self.end_headers()

                    self.wfile.write(json_data.encode("utf-8"))
                    self.wfile.write(b"\n")
                except:
                    break  # this is to make sure that a cut connection doesnt hang

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

            time.sleep(1 / 30)


if __name__ == "__main__":
    # Initialize the video capture
    cap = cv2.VideoCapture("http://192.168.0.127:8000/stream.mjpg")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VID_IN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VID_IN_HEIGHT)

    PORT = 8000
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, VideoStreamHandler)
    print(f"Starting server on port {PORT}...")
    try:
        httpd.serve_forever()
    finally:
        cap.release()
        cv2.destroyAllWindows()
