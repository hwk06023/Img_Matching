import cv2
import numpy as np

point_img1 = cv2.imread('img/point1.png')
point_img2 = cv2.imread('img/point2.png')
point_img3 = cv2.imread('img/point3.png')
point_img4 = cv2.imread('img/point4.png')

video = cv2.VideoCapture('video/InVideos.mp4')

if not video.isOpened():
    print("Could not Open :")
    exit(0)

length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

print("length :", length)
print("width :", width)
print("height :", height)
print("fps :", fps)
