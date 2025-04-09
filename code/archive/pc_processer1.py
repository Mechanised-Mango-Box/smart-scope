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
PAGE = """\
<html>
<head>
<title>PC MJPEG streaming demo</title>
</head>
<body>
<h1>PC MJPEG Streaming Demo</h1>
<img src="stream.mjpg" width="{0}" height="{1}" />
</body>
</html>
""".format(
    VID_OUT_WIDTH, VID_OUT_HEIGHT
)

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


# StreamingOutput Class
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


# StreamingHandler Class
class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(301)
            self.send_header("Location", "/index.html")
            self.end_headers()
        elif self.path == "/index.html":
            content = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", 0)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except Exception as e:
                logging.warning("Removed streaming client %s: %s", self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()


# StreamingServer Class
class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


# Function to detect objects
def getObjects(img, canvas, thres, nms, draw=True, objects=[]):
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
                    cv2.rectangle(canvas, box, color=HUD_GREEN, thickness=HUD_THICKNESS)
                    cv2.putText(canvas, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, HUD_FONT_SIZE, HUD_GREEN, HUD_THICKNESS)
                    cv2.putText(canvas, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, HUD_FONT_SIZE, HUD_GREEN, HUD_THICKNESS)
    return img, objectInfo


if __name__ == "__main__":
    # Initialize the video capture
    cap = cv2.VideoCapture("http://192.168.0.127:8000/stream.mjpg")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VID_IN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VID_IN_HEIGHT)

    # Initialize the streaming output
    output = StreamingOutput()

    # Start the streaming server in a separate thread
    address = ("", 8000)
    server = StreamingServer(address, StreamingHandler)
    server_thread = Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            canvas = np.zeros([VID_IN_WIDTH, VID_IN_HEIGHT, 3], dtype=np.uint8)
            result, objectInfo = getObjects(img, canvas, ACC_THRESHOLD, 0.2)

            # Encode frame as JPEG
            stream = np.flipud(canvas)
            scale_factor = VID_IN_HEIGHT / VID_OUT_HEIGHT
            # Calculate the zoom factor
            zoom_factors = (1 / scale_factor, 1 / scale_factor, 1)  # Last factor is 1 for the color channels
            # Rescale the image using scipy's zoom function
            stream = scipy.ndimage.zoom(stream, zoom_factors, order=3)  # order=3 uses cubic interpolation

            ret, jpeg = cv2.imencode(".jpg", stream)
            if ret:
                output.write(jpeg.tobytes())

            cv2.imshow("Output - CamFeed", result)
            cv2.imshow("Output - Obj ID", canvas)
            cv2.imshow("Output - Stream", stream)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        server.shutdown()
        server.server_close()
