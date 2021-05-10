import cv2
import numpy as np

from utils import RANSAC


def main():
    ########################################### Copied From Q3 ###########################################

    im1 = cv2.imread("../resources/im03.jpg")
    im2 = cv2.imread("../resources/im04.jpg")

    # RGB -> GRAY
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Find interest points
    interest_points_1 = sift.detect(im1_gray, None)
    interest_points_2 = sift.detect(im2_gray, None)

    # Compute Descriptors
    _, descriptor1 = sift.compute(im1_gray, interest_points_1)
    _, descriptor2 = sift.compute(im2_gray, interest_points_2)

    # Use BFMatcher to match points between two images
    bf_matcher = cv2.BFMatcher()
    match_points = bf_matcher.knnMatch(descriptor1, descriptor2, k=2)

    # Ratio between most similar and second most similar should less than 0.9
    final_match_points = []
    for m, n in match_points:
        if m.distance / n.distance < 0.8:
            final_match_points.append(m)

    src_points = []
    dst_points = []
    for match in final_match_points:
        src_points.append(interest_points_2[match.trainIdx].pt)
        dst_points.append(interest_points_1[match.queryIdx].pt)
    ########################################### Copied From Q3 ###########################################

    # Get Homography Matrix form Implemented RANSAC function
    H = RANSAC(src_points, dst_points, threshold=25.0, maxIters=10000)
    print("Homography Matrix: \n", H)

    H[0, 2] = 0
    H[1, 2] = 0
    im2_warp = cv2.warpPerspective(im2, H, (9000, 4500))
    cv2.imwrite('out/res20.jpg', im2_warp)


if __name__ == '__main__':
    main()
