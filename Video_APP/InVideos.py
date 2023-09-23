import cv2
import numpy as np
import matplotlib.pyplot as plt

point_img1 = cv2.imread('Video_APP/img/point_1.png', cv2.IMREAD_GRAYSCALE)
point_img2 = cv2.imread('Video_APP/img/point_2.png', cv2.IMREAD_GRAYSCALE)
'''
point_img3 = cv2.imread('Video_APP/img/point_3.png', cv2.IMREAD_GRAYSCALE)
point_img4 = cv2.imread('Video_APP/img/point_4.png', cv2.IMREAD_GRAYSCALE) '''

video = cv2.VideoCapture('Video_APP/video/InVideos.mp4', cv2.IMREAD_GRAYSCALE)

if not video.isOpened():
    print("Could not Open :")
    exit(0)

detector = cv2.SIFT_create()

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (128, 256))
    point_img1 = cv2.resize(point_img1, (128, 256))
    point_img2 = cv2.resize(point_img2, (128, 256))

    if not ret:
        break

    # SIFT 특징점 검출 및 매칭
    kp, des = detector.detectAndCompute(frame, None)
    kp1, des1 = detector.detectAndCompute(point_img1, None)
    kp2, des2 = detector.detectAndCompute(point_img2, None)
    '''
    kp3, des3 = detector.detectAndCompute(point_img3, None)
    kp4, des4 = detector.detectAndCompute(point_img4, None)
    '''
    # 특징점 매칭
    bf = cv2.BFMatcher()
    matches1 = bf.knnMatch(des, des1, k=2)
    matches2 = bf.knnMatch(des, des2, k=2)
    '''
    matches3 = bf.knnMatch(des, des3, k=2)
    matches4 = bf.knnMatch(des, des4, k=2) '''

    # 거리가 가까운 매칭 결과 선택
    good_matches = []
    ratio = 0.7
    good_matches.append(list(first for first,second in matches1 \
                    if first.distance < second.distance * ratio))
    
    good_matches.append(list(first for first,second in matches2 \
                    if first.distance < second.distance * ratio))
    
    '''
    good_matches.append(first for first,second in matches3 \
                    if first.distance < second.distance * ratio)
    
    good_matches.append(first for first,second in matches4 \
                    if first.distance < second.distance * ratio) '''
    
    '''
    print(len(good_matches[2]),len(matches3))
    print(len(good_matches[3]),len(matches4)) '''

    if len(good_matches[0]) <= 5 and len(good_matches[1]) <= 5:
        print('Cannot find Homography')
        continue
    
    elif len(good_matches[0]) > 5 and len(good_matches[1]) > 5:
        src_pts1 = np.float32([ kp[m.queryIdx].pt for m in good_matches[0] ])
        src_pts2 = np.float32([ kp[m.queryIdx].pt for m in good_matches[1] ])

        dst_pts1 = np.float32([ kp1[m.trainIdx].pt for m in good_matches[0] ])
        dst_pts2 = np.float32([ kp2[m.trainIdx].pt for m in good_matches[1] ])

        mtrx1, mask1 = cv2.findHomography(src_pts1, dst_pts1)
        mtrx2, mask2 = cv2.findHomography(src_pts2, dst_pts2)

        h,w, = frame.shape[:2]
        pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
        dst1 = cv2.perspectiveTransform(pts,mtrx1)
        dst2 = cv2.perspectiveTransform(pts,mtrx2)

        point_img1 = cv2.polylines(point_img1,[np.int32(dst1)],True,255,3, cv2.LINE_AA)
        point_img2 = cv2.polylines(point_img2,[np.int32(dst2)],True,255,3, cv2.LINE_AA)

        res1 = cv2.drawMatches(frame, kp, point_img1, kp1, good_matches, None, \
                            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        res2 = cv2.drawMatches(frame, kp, point_img2, kp2, good_matches, None, \
                            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(res1)
        plt.imshow(res2)
        plt.axis('off')
        plt.show()

    else:
        print('Find Homography')

        print('-- good matches --')
        print(len(matches1))
        print(len(matches2))
        print(len(good_matches[0]))
        print(len(good_matches[1]))

        if len(good_matches[0]) <= 5:
            src_pts = np.float32([ kp[m.queryIdx].pt for m in good_matches[1] ])
            dst_pts = np.float32([ kp1[m.trainIdx].pt for m in good_matches[1] ])
        elif len(good_matches[1]) <= 5:
            src_pts = np.float32([ kp[m.queryIdx].pt for m in good_matches[0] ])
            dst_pts = np.float32([ kp1[m.trainIdx].pt for m in good_matches[0] ])

        mtrx, mask = cv2.findHomography(src_pts, dst_pts)

        h,w, = frame.shape[:2]

        pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
        dst = cv2.perspectiveTransform(pts,mtrx)

        point_img1 = cv2.polylines(point_img1,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        res = cv2.drawMatches(frame, kp, point_img1, kp1, good_matches[0], None, \
                            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
            
        plt.figure(figsize=(15, 10))
        plt.imshow(res)
        plt.axis('off')
        plt.show()

video.release()
