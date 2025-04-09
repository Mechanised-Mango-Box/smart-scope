# Steps
## 1. 3D Print
## 2. Pi Setup
- ssh make it faster
- if wifi doesnt work: https://forums.raspberrypi.com/viewtopic.php?t=201850
	- add `country=AU` to `sudo nano /etc/wpa_supplicant/wpa_supplicant.conf
`
## 3. Display
https://www.waveshare.com/wiki/2.4inch_LCD_Module
- use Raspberry Pi Pinout
- image uses: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.transpose
## 4. Camera
- dri2 error - enable full kms opengl driver
## 5. OpenCV
### (a) Open CV on Pi Zero W 2
- Memory:
	- Change gpu memory
		- https://raspberrypi.stackexchange.com/questions/673/what-is-the-optimum-split-of-main-versus-gpu-memory
		- Default = 128
		- new = 256?? - 50%/50% with cpu
			- should be fine since video/opencv is gpu bound
	- DONT TOUCH SWAP: https://raspberrypi.stackexchange.com/questions/88643/pi-zero-w-swap-size-limit
		- will cause crash :(
		- only change gpu
- https://github.com/raspberrypi/picamera2/blob/main/examples/opencv_face_detect.py
	- https://github.com/raspberrypi/picamera2/tree/main/examples
	- https://stackoverflow.com/questions/7106012/download-a-single-folder-or-directory-from-a-github-repository
- Too big to work???
### (b) Ping Pong (opencv on pc)
- https://www.switchdoc.com/2019/07/mjpeg-mjpg-python-streaming-overlay/
- https://github.com/raspberrypi/picamera2/blob/main/examples/mjpeg_server.py
	- Stream at: `http://<ip>:<port>/stream.mjpg`

#### `mjpeg` to `opencv` to `mjpeg`
- slow
- adjusting res is hard
- pi cant handle data

#### `mjpeg` to `opencv` to `json`
- faster
	- allows leverage of pc power
		- cuda???
		- save pi battery life/ power draw
	- can be wireless
		- headless w/ ssh?
		- demo with vlc
- allows monitoring
