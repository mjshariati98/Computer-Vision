import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    src_image = cv2.imread('../resources/vns.jpg')
    height, width = src_image.shape[:2]

    # y = ax + b , h: (a,b)
    h = (-0.04848260554402862, 2802.5212851358565)

    K = np.array([
        [1.26353445e+04, 0.00000000e+00, 3.11609853e+03],
        [0.00000000e+00, 1.26353445e+04, 1.25001591e+03],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    R = Rotation.from_euler('YZX', [0, 2.7587845052519064, 6.329000672777569], degrees=True).as_matrix()

    H = K @ R @ np.linalg.inv(K)
    print("Homography Matrix: ")
    print(H, "\n")

    min_x, min_y, max_x, max_y = get_projection_of_vertices(width, height, H)
    max_x = int(max_x - min_x)
    max_y = int(max_y - min_y)

    T = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ])

    warped_img = cv2.warpPerspective(src_image, T @ H, (max_x, max_y))
    cv2.imwrite("out/res04.jpg", warped_img)

    # with horizon line
    cv2.line(src_image, (0, int(h[1])), (width, int(h[0] * width + h[1])), color=(0, 0, 255), thickness=3)
    warped_img_horizon = cv2.warpPerspective(src_image, T @ H, (max_x, max_y))

    cv2.imwrite("out/res04-horizon-line.jpg", warped_img_horizon)


def copyof(image):
    copy = np.zeros(image.shape)
    copy[:, :, :] = image[:, :, :]
    return copy


def get_projection_of_vertices(width, height, homography):
    p1 = np.array([[0],
                   [0],
                   [1]])
    p2 = np.array([[width],
                   [0],
                   [1]])
    p3 = np.array([[0],
                   [height],
                   [1]])
    p4 = np.array([[width],
                   [height],
                   [1]])

    p1_warped = homography @ p1
    p1_warped = p1_warped / p1_warped[2, 0]
    p2_warped = homography @ p2
    p2_warped = p2_warped / p2_warped[2, 0]
    p3_warped = homography @ p3
    p3_warped = p3_warped / p3_warped[2, 0]
    p4_warped = homography @ p4
    p4_warped = p4_warped / p4_warped[2, 0]

    min_x = min(p1_warped[0, 0], p2_warped[0, 0], p3_warped[0, 0], p4_warped[0, 0])
    min_y = min(p1_warped[1, 0], p2_warped[1, 0], p3_warped[1, 0], p4_warped[1, 0])

    max_x = max(p1_warped[0, 0], p2_warped[0, 0], p3_warped[0, 0], p4_warped[0, 0])
    max_y = max(p1_warped[1, 0], p2_warped[1, 0], p3_warped[1, 0], p4_warped[1, 0])

    return min_x, min_y, max_x, max_y


if __name__ == '__main__':
    main()
