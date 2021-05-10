import cv2
import numpy as np
import random

from utils import show_together, copyof, draw_lines_on_together_image


def main():
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

    # Show points on image
    im1_with_interest_points = cv2.drawKeypoints(im1, interest_points_1, copyof(im1), color=(0, 255, 0))
    im2_with_interest_points = cv2.drawKeypoints(im2, interest_points_2, copyof(im2), color=(0, 255, 0))

    corners = show_together(im1_with_interest_points, im2_with_interest_points)
    cv2.imwrite('out/res13_corners.jpg', corners)

    # Compute Descriptors
    _, descriptor1 = sift.compute(im1_gray, interest_points_1)
    _, descriptor2 = sift.compute(im2_gray, interest_points_2)

    # Use BFMatcher to match points between two images
    bf_matcher = cv2.BFMatcher()
    match_points = bf_matcher.knnMatch(descriptor1, descriptor2, k=2)

    # Ratio between most similar and second most similar should less than 0.9
    final_match_points = []
    for m, n in match_points:
        if m.distance / n.distance < 0.90:
            final_match_points.append(m)

    final_match_points1 = []  # just to draw following image
    final_match_points2 = []
    for match_point in final_match_points:
        final_match_points1.append(interest_points_1[match_point.queryIdx])
        final_match_points2.append(interest_points_2[match_point.trainIdx])

    # Show points on image
    im1_with_match_points = cv2.drawKeypoints(im1_with_interest_points, final_match_points1,
                                              copyof(im1_with_interest_points), color=(255, 0, 0))
    im2_with_match_points = cv2.drawKeypoints(im2_with_interest_points, final_match_points2,
                                              copyof(im2_with_interest_points), color=(255, 0, 0))

    match_points_together = show_together(im1_with_match_points, im2_with_match_points)
    cv2.imwrite('out/res14_correspondences.jpg', match_points_together)

    # show match lines
    match_points_together_line = cv2.drawMatches(im1_with_match_points, interest_points_1, im2_with_match_points,
                                                 interest_points_2, final_match_points, None, matchColor=(255, 0, 0),
                                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imwrite('out/res15_matches.jpg', match_points_together_line)

    # Show only 20 random matches
    random_matches = random.sample(final_match_points, 20)
    match_points_together_line = cv2.drawMatches(im1_with_match_points, interest_points_1, im2_with_match_points,
                                                 interest_points_2, random_matches, None, matchColor=(255, 0, 0),
                                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imwrite('out/res16.jpg', match_points_together_line)

    # Find Homography
    # Get Src(im2) and Dst(im1) match points for cv2.findHomography function
    src_points = []
    dst_points = []
    for match in final_match_points:
        src_points.append(interest_points_2[match.trainIdx].pt)
        dst_points.append(interest_points_1[match.queryIdx].pt)
    src_points = np.float32(src_points).reshape(-1, 1, 2)
    dst_points = np.float32(dst_points).reshape(-1, 1, 2)

    # Run RANSAC algorithm with threshold = 5.0
    # mask specifies that which points are inlier and which are outlier
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 150.0)
    print("Homography Matrix: \n", H)

    inlier_matches = cv2.drawMatches(im1_with_match_points, interest_points_1, im2_with_match_points, interest_points_2,
                                     final_match_points, None, matchColor=(0, 0, 255),
                                     matchesMask=mask.ravel().tolist(),
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('out/res17.jpg', inlier_matches)

    # MisMatch Example
    mismatch_example = draw_lines_on_together_image(im1, im2, (1144, 1308), (886, 986))
    cv2.imwrite("out/res18_mismatch.jpg", mismatch_example)

    # # calculate min_x and min_y after warp
    # height, width = im2.shape[0], im2.shape[1]
    # src = np.array([[0, width, width, 0],
    #                 [0, 0, height, height],
    #                 [1, 1, 1, 1]])
    # dst = np.matmul(H, src)
    # # divide first and second rows by third row
    # dst[0, 0] = dst[0, 0] / dst[2, 0]
    # dst[0, 1] = dst[0, 1] / dst[2, 1]
    # dst[0, 2] = dst[0, 2] / dst[2, 2]
    # dst[0, 3] = dst[0, 3] / dst[2, 3]
    # dst[1, 0] = dst[1, 0] / dst[2, 0]
    # dst[1, 1] = dst[1, 1] / dst[2, 1]
    # dst[1, 2] = dst[1, 2] / dst[2, 2]
    # dst[1, 3] = dst[1, 3] / dst[2, 3]
    # print(dst)
    # # smallest in first row = -3.88827442e+03   -> min_x = 3888
    # # smallest in second row = -2.15784451e+03  -> min_y = 2157

    # Change Homography Matrix to warp correctly
    H[0, 2] = H[0, 2] + 3888
    H[1, 2] = H[1, 2] + 2157

    im2_warp = cv2.warpPerspective(im2, H, (6000, 4500))
    cv2.imwrite('out/res19.jpg', im2_warp)


if __name__ == '__main__':
    main()
