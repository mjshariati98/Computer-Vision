import numpy as np
import random
from scipy.linalg import svd
import sys
import math


def RANSAC(src_points, dst_points, threshold, p=0.99, maxIters=10000):
    s = 4  # Minimum points
    number_of_points = len(src_points)

    counter = 0
    w_min = 0
    N = sys.maxsize
    best_H = None

    while N > counter and counter < maxIters:
        random_indices = random.sample(range(0, number_of_points), s)  # select s random index

        random_src_points = [src_points[i] for i in random_indices]
        random_dst_points = [dst_points[i] for i in random_indices]

        A = getA(random_src_points, random_dst_points)
        U, S, V_T = svd(A)
        H = V_T[-1].reshape(3, 3)

        supports = 0  # number of inliers by this H
        for i in range(number_of_points):
            src = np.array([[src_points[i][0]],
                            [src_points[i][1]],
                            [1]])
            dst_point_by_H = np.matmul(H, src)

            dst_point_by_H[0] /= dst_point_by_H[2]
            dst_point_by_H[1] /= dst_point_by_H[2]

            diff = math.sqrt(
                ((dst_points[i][0] - dst_point_by_H[0]) ** 2) + ((dst_points[i][1] - dst_point_by_H[1]) ** 2))
            if diff < threshold:
                supports += 1

            w = supports / number_of_points
            if w > w_min:
                w_min = w
                N = math.log(1 - p) / math.log(1 - (w ** s))
                best_H = H

        counter += 1
        print(counter)

    inliers_idx = []
    for i in range(number_of_points):
        src = np.array([[src_points[i][0]],
                        [src_points[i][1]],
                        [1]])
        dst_point_by_H = np.matmul(best_H, src)

        dst_point_by_H[0] /= dst_point_by_H[2]
        dst_point_by_H[1] /= dst_point_by_H[2]

        diff = math.sqrt(
            ((dst_points[i][0] - dst_point_by_H[0]) ** 2) + ((dst_points[i][1] - dst_point_by_H[1]) ** 2))
        if diff < threshold:
            inliers_idx.append(i)

    src_inlier = [src_points[i] for i in inliers_idx]
    dst_inlier = [dst_points[i] for i in inliers_idx]

    A = getA(src_inlier, dst_inlier)
    U, S, V_T = svd(A)
    H = V_T[-1].reshape(3, 3)

    return H


def getA(src_points, dst_points):
    A = np.zeros((2 * len(src_points), 9))
    for i in range(len(src_points)):
        A[2 * i:2 * i + 2, :] = np.array([[-src_points[i][0], -src_points[i][1], -1, 0, 0, 0,
                                           src_points[i][0] * dst_points[i][0], src_points[i][1] * dst_points[i][0],
                                           dst_points[i][0]],
                                          [0, 0, 0, -src_points[i][0], -src_points[i][1], -1,
                                           src_points[i][0] * dst_points[i][1], src_points[i][1] * dst_points[i][1],
                                           dst_points[i][1]]])
    return A
