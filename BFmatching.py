import cv2

image1 = cv2.imread('navi2_img.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('detected2_3.png', cv2.IMREAD_GRAYSCALE)
flag = 0

orb = cv2.SIFT_create()

keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = matcher.match(descriptors1, descriptors2)

matches = sorted(matches, key=lambda x: x.distance)

threshold_distance = 50

if len(matches) > threshold_distance:
    print("True - Images have matching features.")
    flag=1
else:
    print("False - Images do not have matching features.")

# Draw matches
match_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('Matches', match_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
