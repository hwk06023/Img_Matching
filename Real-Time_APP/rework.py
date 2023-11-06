import cv2
import numpy as np
import matplotlib.pyplot as plt
import pygame
import threading
import sys
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore

# Using PYQT
app = QtWidgets.QApplication([])
win = QtWidgets.QWidget()
vbox = QtWidgets.QVBoxLayout()
label = QtWidgets.QLabel()

running = False
def run():
    global running
    cap = cv2.VideoCapture(-1)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    label.resize(width, height)
    while running:
        ret, img = cap.read()
        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            h,w,c = img.shape
            qImg = QtGui.QImage(img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qImg)
            label.setPixmap(pixmap)
        else:
            QtWidgets.QMessageBox.about(win, "Error", "Cannot read frame.")
            print("cannot read frame.")
            break
    cap.release()
    print("Thread end.")

def stop():
    global running
    running = False
    print("stoped..")

def start():
    global running
    running = True
    th = threading.Thread(target=run)
    th.start()
    print("started..")

def onExit():
    print("exit")
    stop()

# if you build in window
'''
import winsound as sd

def beepsound():
    fr = 2000    # range : 37 ~ 32767
    du = 1000     # 1000 ms == 1 second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)
    
'''

pygame.mixer.pre_init(44100,-16,2,512)
pygame.mixer.init()
matching_sound = pygame.mixer.Sound('Real-Time_APP/sounds/supershy.ogg')
matching_sound.set_volume(0.5)

pointImg_li = []

# default is 0
device = 0
video = cv2.VideoCapture(device)

if not video.isOpened():
    print("Could not Open :")
    exit(0)

video_size = (512, 1024)

video.set(cv2.CAP_PROP_FRAME_WIDTH, video_size[0])
video.set(cv2.CAP_PROP_FRAME_HEIGHT, video_size[1])

fps = video.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = './' + str(resize_frame_size) + '_demo.mp4'
out = cv2.VideoWriter(f'{video_out}', fourcc, fps, (video_size[1], video_size[0]))

detector = cv2.SIFT_create()

total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
count_frames = 0

maching_frames = []

while True:
    print('frame :', count_frames, '/', total_frames)
    maching_frames.append(count_frames)
    count_frames += 1

    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (resize_frame_size, resize_frame_size))

    kp, des = detector.detectAndCompute(frame, None) # kp des
    kp1, des1 = detector.detectAndCompute(point_img1, None) 
    kp2, des2 = detector.detectAndCompute(point_img2, None)

    bf = cv2.BFMatcher()
    matches1 = bf.knnMatch(des, des1, k=2) # k
    matches2 = bf.knnMatch(des, des2, k=2)

    good_matches = []
    ratio = 0.5
    good_matches.append(list(first for first,second in matches1 \
                    if first.distance < second.distance * ratio))
    
    good_matches.append(list(first for first,second in matches2 \
                    if first.distance < second.distance * ratio))

    if len(good_matches[0]) <= 5 and len(good_matches[1]) <= 5: 
        print('Cannot find Homography')
        frame = np.pad(frame, [(0, video_size[0]-resize_frame_size), (0, video_size[1]-resize_frame_size)], mode='constant')
        frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
        out_frame = np.array(frame).reshape(frame.shape[0], frame.shape[1] , 3)
        cv2.imshow("Video_Frame", out_frame)
        cv2.waitKey(1)
        cv2.destroyAllWindows()

        out.write(out_frame)
        continue
    
    else:
        if not pygame.mixer.music.get_busy():
            matching_sound.play()

        print('Find Homography')

        print('-- good matches --')
        print('total matches :', len(matches1))

        # point_num
        for i in range(2):
            print(i,'- matches :',len(good_matches[i]))

            if len(good_matches[i]) > 5:
                if i == 0:
                    kp_point = kp1
                    matches_point = matches1
                    good_matches_point = good_matches[0]
                    point = point_img1
                elif i == 1:
                    kp_point = kp2
                    matches_point = matches2
                    good_matches_point = good_matches[1]
                    point = point_img2

                src_pts = np.float32([ kp[m.queryIdx].pt for m in good_matches[i] ]) # queryIdx, trainIdx 
                dst_pts = np.float32([ kp_point[m.trainIdx].pt for m in good_matches[i] ])

                break

        mtrx, mask = cv2.findHomography(src_pts, dst_pts)
        
        h,w = frame.shape[:2]

        pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
        dst = cv2.perspectiveTransform(pts,mtrx)

        point = cv2.polylines(point,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        res = cv2.drawMatches(frame, kp, point, kp_point, good_matches_point, None, \
                            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        
        maching_frames.append(res)
        
        h, w = res.shape[:2]
        if (video_size[0]-h) * (video_size[1]-w) != 0:
          res = np.pad(res, [(0, video_size[0]-h), (0, video_size[1]-w)], mode='constant')

        out_res = np.array(res).reshape(res.shape[0], res.shape[1], 3)
        out.write(out_res)

        cv2.imshow("Video_Frame", res)
        cv2.destroyAllWindows()

out.release()
cv2.VideoCapture.release()