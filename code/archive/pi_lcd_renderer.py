from PIL import Image
import numpy as np
import requests
from io import BytesIO

# Replace with your IP webcam's MJPEG stream URL
stream_url = "http://localhost:8000/stream.mjpg"

def stream_ip_camera(url):
    stream = requests.get(url, stream=True)
    bytes_data = bytes()

    for chunk in stream.iter_content(chunk_size=1024):
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')  # Start of JPEG
        b = bytes_data.find(b'\xff\xd9')  # End of JPEG
        
        if a != -1 and b != -1:
            jpg = bytes_data[a:b + 2]
            bytes_data = bytes_data[b + 2:]

            # Convert bytes data to an image using PIL
            img = Image.open(BytesIO(jpg))
            
            # show here

if __name__ == "__main__":
    stream_ip_camera(stream_url)