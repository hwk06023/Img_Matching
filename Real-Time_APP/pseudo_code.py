import cv2
import numpy as np
import matplotlib.pyplot as plt
import pygame

# if you build in window
'''
import winsound as sd

def beepsound():
    fr = 2000    # range : 37 ~ 32767
    du = 1000     # 1000 ms == 1 second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)
    
'''

# if you build in linux or MacOS
pygame.mixer.pre_init(44100,-16,2,512) # frequency=44100, size=-16, channels=2, buffer=512
pygame.mixer.init()
matching_sound = pygame.mixer.Sound('Real-Time_APP/sounds/supershy.ogg') # Sound file path
matching_sound.set_volume(0.5)

point_img1 = cv2.imread('Real-Time_APP/img/point_1.png', cv2.IMREAD_GRAYSCALE) # image file path    
point_img2 = cv2.imread('Real-Time_APP/img/point_2.png', cv2.IMREAD_GRAYSCALE) # image file path

# The performence of 1024 x 1024 version is better than 512 x 512 version but I think it's too slow
point_img1 = cv2.resize(point_img1, (512, 512))
point_img2 = cv2.resize(point_img2, (512, 512))

detector = cv2.SIFT_create()
bf = cv2.BFMatcher()

# kp : keypoint, des : descriptor 
kp1, des1 = detector.detectAndCompute(point_img1, None) 
kp2, des2 = detector.detectAndCompute(point_img2, None)

# default is 0, if you have another camera, change to proper number or path
device = 0
video = cv2.VideoCapture(device)

if not video.isOpened():
    print("Could not Open :")
    exit(0)

video_size = (512, 1024)

video.set(cv2.CAP_PROP_FRAME_WIDTH, video_size[0])
video.set(cv2.CAP_PROP_FRAME_HEIGHT, video_size[1])

# set video output
fps = video.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = './demo.mp4'
out = cv2.VideoWriter(f'{video_out}', fourcc, fps, (video_size[1], video_size[0]))

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
    frame = cv2.resize(frame, (512, 512))

    kp, des = detector.detectAndCompute(frame, None) 

    matches1 = bf.knnMatch(des, des1, k=2) # k : 아래 DMatch 개수
    matches2 = bf.knnMatch(des, des2, k=2)
    # matches : [[<DMatch 000001F1F4F1F7B0>, <DMatch 000001F1F4F1F7F0>], [<DMatch 000001F1F4F1F>, <DMatch 000001F1F4F1F>], ...]

    good_matches = []
    ratio = 0.5
    good_matches.append(list(first for first,second in matches1 \
                    if first.distance < second.distance * ratio))
    
    good_matches.append(list(first for first,second in matches2 \
                    if first.distance < second.distance * ratio))

    if len(good_matches[0]) <= 5 and len(good_matches[1]) <= 5: 
        print('Cannot find good matches')
        frame = np.pad(frame, [(0, video_size[0]-512), (0, video_size[1]-512)], mode='constant')
        frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
        out_frame = np.array(frame).reshape(frame.shape[0], frame.shape[1] , 3)
        cv2.imshow("Video_Frame", out_frame)
        cv2.waitKey(1)
        cv2.destroyAllWindows()

        out.write(out_frame)
        continue
    
    else:
        # play sound
        if not pygame.mixer.music.get_busy():
            matching_sound.play()
        
        #print good matches
        print('Find good matches, Matching successfully')

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

                # queryIdx : navigt's index, trainIdx : pointImg's index
                src_pts = np.float32([ kp[m.queryIdx].pt for m in good_matches[i] ]) 
                dst_pts = np.float32([ kp_point[m.trainIdx].pt for m in good_matches[i] ])

                break
        
        # find Homography
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

        # Save video
        out_res = np.array(res).reshape(res.shape[0], res.shape[1], 3)
        out.write(out_res)

        # Display
        cv2.imshow("Video_Frame", res)
        cv2.destroyAllWindows()

out.release()
cv2.VideoCapture.release()