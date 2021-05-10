import cv2
import numpy as np


def main():
    logo = cv2.imread("../resources/logo.png")

    R = np.array([
        [1, 0, 0],
        [0, 0.52999894, 0.847998304],
        [0, -0.847998304, 0.52999894]
    ])

    C = np.array([
        [0],
        [-40],
        [0]
    ])

    t = - np.matmul(R, C)

    k1 = np.array([
        [500, 0, 280],
        [0, 500, 2390],
        [0, 0, 1]
    ])

    k2 = np.array([
        [500, 0, 128],
        [0, 500, 128],
        [0, 0, 1]
    ])

    n_T = np.array([[0, 0, -1]])
    d = 25

    H = np.matmul(np.matmul(k2, R - (np.matmul(t, n_T) / d)), np.linalg.inv(k1))
    H_inverse = np.linalg.inv(H)
    print(H)
    print(H_inverse)

    im_dst = cv2.warpPerspective(logo, H_inverse, (560, 1130))
    cv2.imwrite("out/res12.jpg", im_dst)


if __name__ == '__main__':
    main()
