import cv2
import numpy as np
import matplotlib.pyplot as plt

# point_num = int(input('Input Point Number : '))
# I'll use list for point_img and good_matches
# After 4 point, you can add more point

point_img1 = cv2.imread('Video_APP/img/point_1.png', cv2.IMREAD_GRAYSCALE)
point_img2 = cv2.imread('Video_APP/img/point_2.png', cv2.IMREAD_GRAYSCALE)
# point_img3 = cv2.imread('Video_APP/img/point3_1.png', cv2.IMREAD_GRAYSCALE)
# point_img4 = cv2.imread('Video_APP/img/point_4.png', cv2.IMREAD_GRAYSCALE)

# If you want to resize frame size, change this value
resize_frame_size = 256
query_img_width = 256

h, w = 256, 256
point_img1 = cv2.resize(point_img1, (query_img_width, query_img_width * h // w))
max_height = max(256, query_img_width * h // w)
point_img2 = cv2.resize(point_img2, (query_img_width, query_img_width * h // w))
max_height = max(max_height, query_img_width * h // w)
# h, w = point_img3.shape
# point_img3 = cv2.resize(point_img3, (query_img_width, query_img_width * h // w))
# max_height = max(max_height, query_img_width * h // w)
# h, w = point_img4.shape
# point_img4 = cv2.resize(point_img4, (query_img_width, query_img_width * h // w))
# max_height = max(max_height, query_img_width * h // w)

video = cv2.VideoCapture('Video_APP/video/InVideos.mp4')

video_size = (max_height, resize_frame_size+query_img_width)

fps = video.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = './' + str(resize_frame_size) + '_demo.mp4'
out = cv2.VideoWriter(f'{video_out}', fourcc, fps, (video_size[1], video_size[0]))

if not video.isOpened():
    print("Could not Open :")
    exit(0)

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
    frame = cv2.resize(frame, (256, 256))

    kp, des = detector.detectAndCompute(frame, None)
    kp1, des1 = detector.detectAndCompute(point_img1, None)
    kp2, des2 = detector.detectAndCompute(point_img2, None)
    # kp3, des3 = detector.detectAndCompute(point_img3, None)
    # kp4, des4 = detector.detectAndCompute(point_img4, None)

    bf = cv2.BFMatcher()
    matches1 = bf.knnMatch(des, des1, k=2)
    matches2 = bf.knnMatch(des, des2, k=2)
    # matches3 = bf.knnMatch(des, des3, k=2)
    # matches4 = bf.knnMatch(des, des4, k=2) 

    good_matches = []
    ratio = 0.5
    good_matches.append(list(first for first,second in matches1 \
                    if first.distance < second.distance * ratio))
    
    good_matches.append(list(first for first,second in matches2 \
                    if first.distance < second.distance * ratio))
    ''' 
    good_matches.append(list(first for first,second in matches3 \
                    if first.distance < second.distance * ratio)) '''
    '''
    good_matches.append(list(first for first,second in matches4 \
                    if first.distance < second.distance * ratio)) '''

    if len(good_matches[0]) <= 5 and len(good_matches[1]) <= 5: # len(good_matches[2]) <= 5:
        print('Cannot find Homography')
        frame = np.pad(frame, [(0, video_size[0]-resize_frame_size), (0, video_size[1]-resize_frame_size)], mode='constant')
        frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
        out.write(np.array(frame).reshape(frame.shape[0], frame.shape[1] , 3))
        continue
    
    else:
        print('Find Homography')

        print('-- good matches --')
        print('total matches :', len(matches1))

        # 4 -> point_num
        for i in range(2):
            print(i,'- matches :',len(good_matches[i]))
        for i in range(2):
            if len(good_matches[i]) > 5:
                if i == 0:
                    kp_point = kp1
                    matches_point = matches1
                    good_matches_point = good_matches[0]
                    point = point_img1
                elif i == 1 and len(good_matches[1]) > len(good_matches[0]):
                    kp_point = kp2
                    matches_point = matches2
                    good_matches_point = good_matches[1]
                    point = point_img2
                '''
                elif i == 2 and len(good_matches[2]) > len(good_matches[0]) and len(good_matches[2]) > len(good_matches[1]):
                    kp_point = kp3
                    matches_point = matches3
                    good_matches_point = good_matches[2]
                    point = point_img3'''
                
                src_pts = np.float32([ kp[m.queryIdx].pt for m in good_matches[i] ])
                dst_pts = np.float32([ kp_point[m.trainIdx].pt for m in good_matches[i] ])

                break

        '''
        if len(good_matches[0]) > 5:
            src_pts = np.float32([ kp[m.queryIdx].pt for m in good_matches[0] ])
            dst_pts = np.float32([ kp1[m.trainIdx].pt for m in good_matches[0] ])
        elif len(good_matches[1]) > 5:
            src_pts = np.float32([ kp[m.queryIdx].pt for m in good_matches[1] ])
            dst_pts = np.float32([ kp1[m.trainIdx].pt for m in good_matches[1] ])
        elif len(good_matches[2]) > 5:
            src_pts = np.float32([ kp[m.queryIdx].pt for m in good_matches[2] ])
            dst_pts = np.float32([ kp1[m.trainIdx].pt for m in good_matches[2] ])
        elif len(good_matches[3]) > 5:
            src_pts = np.float32([ kp[m.queryIdx].pt for m in good_matches[3] ])
            dst_pts = np.float32([ kp1[m.trainIdx].pt for m in good_matches[3] ]) '''

        mtrx, mask = cv2.findHomography(src_pts, dst_pts)
        
        h,w = frame.shape[:2]

        pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
        dst = cv2.perspectiveTransform(pts,mtrx)

        point = cv2.polylines(point,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        res = cv2.drawMatches(frame, kp, point, kp_point, good_matches_point, None, \
                            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        
        h, w = res.shape[:2]
        if (video_size[0]-h) * (video_size[1]-w) != 0:
          res = np.pad(res, [(0, video_size[0]-h), (0, video_size[1]-w)], mode='constant')
        out.write(np.array(res).reshape(res.shape[0], res.shape[1], 3))

        '''
        plt.figure(figsize=(15, 10))
        plt.imshow(res)
        plt.axis('off')
        plt.show() '''

out.release()



# 68 line

'''
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
    plt.show() '''

'''
Can I loop for variable?

'''