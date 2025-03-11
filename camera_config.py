# camera_config.py
import cv2

def init_video_capture():
    videocapture=cv2.VideoCapture(0)
    return videocapture

def release_video_capture(videocapture):
    videocapture.release()