import cv2
import numpy as np
import matplotlib.pyplot as plt

point_img1 = cv2.imread('img/point1.png', cv2.IMREAD_GRAYSCALE)
point_img2 = cv2.imread('img/point2.png', cv2.IMREAD_GRAYSCALE)
point_img3 = cv2.imread('img/point3.png', cv2.IMREAD_GRAYSCALE)
point_img4 = cv2.imread('img/point4.png', cv2.IMREAD_GRAYSCALE)

video = cv2.VideoCapture('video/InVideos.mp4', cv2.IMREAD_GRAYSCALE)

if not video.isOpened():
    print("Could not Open :")
    exit(0)

detector = cv2.SIFT_create()

frameCounter = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    # SIFT 특징점 검출 및 매칭
    kp, des = detector.detectAndCompute(frame, None)
    kp1, des1 = detector.detectAndCompute(point_img1, None)
    kp2, des2 = detector.detectAndCompute(point_img2, None)
    kp3, des3 = detector.detectAndCompute(point_img3, None)
    kp4, des4 = detector.detectAndCompute(point_img4, None)

    # 특징점 매칭
    bf = cv2.BFMatcher()
    matches1 = bf.knnMatch(des, des1, k=2)
    matches2 = bf.knnMatch(des, des2, k=2)
    '''
    matches3 = bf.knnMatch(des, des3, k=2)
    matches4 = bf.knnMatch(des, des4, k=2) '''

    # 거리가 가까운 매칭 결과 선택
    good_matches = []
    ratio = 0.6
    good_matches.append(first for first,second in matches1 \
                    if first.distance < second.distance * ratio)
    
    good_matches.append(first for first,second in matches2 \
                    if first.distance < second.distance * ratio)
    
    '''
    good_matches.append(first for first,second in matches3 \
                    if first.distance < second.distance * ratio)
    
    good_matches.append(first for first,second in matches4 \
                    if first.distance < second.distance * ratio) '''
    
    print('-- good matches --')
    print(len(good_matches[0]),len(matches1))
    print(len(good_matches[1]),len(matches2))
    '''
    print(len(good_matches[2]),len(matches3))
    print(len(good_matches[3]),len(matches4)) '''

    resultImg1 = cv2.drawMatches(frame, kp, point_img1, kp1, good_matches[0], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    resultImg2 = cv2.drawMatches(frame, kp, point_img2, kp2, good_matches[1], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.figure(figsize=(15, 10))
    plt.imshow(resultImg1)
    plt.imshow(resultImg2)
    plt.axis('off')
    plt.show()

video.release()
