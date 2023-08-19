import cv2

img1 = cv2.imread('./img/navi2_img.png')
img2 = cv2.imread('./img/navi2_temp3.png')

using_FAST = False

feature_detector = dict(SIFT = cv2.SIFT_create(), 
                        ORB = cv2.ORB_create(),
                        brisk = cv2.BRISK_create(),
                        akaze = cv2.AKAZE_create())

matching_type = 'bruteforce'

for type in list(feature_detector.keys()):
    # SIFT 생성
    detector = feature_detector[type]

    if using_FAST == True :
        fast = cv2.FastFeatureDetector_create()
        kp1 = fast.detect(img1, None)
        kp2 = fast.detect(img2, None)
        kp1, des1 = detector.compute(img1, kp1)
        kp2, des2 = detector.compute(img2, kp2)
    else:
        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2, None)

    if matching_type == 'bruteforce':
        matcher = cv2.BFMatcher()
    elif matching_type == 'flann':
        if type == 'SIFT' or type == 'FAST':
            # SIFT 파라미터
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
        elif type == 'ORB' :
            # ORB 파라미터
            FLANN_INDEX_LSH = 6
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                            table_number = 6,
                            key_size = 12,
                            multi_probe_level = 1)
            search_params=dict(checks=32)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x:x.distance)
    good_matches = matches[:30]
    
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
    cv2.imshow("Matching result", img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    matched_points1 = []
    matched_points2 = []
    for match in good_matches:
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        matched_points1.append((int(pt1[0]), int(pt1[1])))
        matched_points2.append((int(pt2[0]), int(pt2[1])))

    print("Matched Points in Image1:", matched_points1)
    print("Matched Points in Image2:", matched_points2)