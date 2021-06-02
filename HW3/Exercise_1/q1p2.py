import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def main():
    src_image = cv2.imread('../resources/vns.jpg')

    V_x = (8249.610434476259, 2402.5586765492417)
    V_y = (-28644.19538099924, 4191.266510918929)
    V_z = (-2407.080058669275, -112670.81249853193)

    a1, b1 = V_x
    a2, b2 = V_y
    a3, b3 = V_z

    p_y = ((a1 * (a2 - a3)) + (b1 * (b2 - b3))) - (((a2 - a3) * a2 * (a1 - a3)) / (a1 - a3)) - (((a2 - a3) * b2 * (b1 - b3)) / (a1 - a3))
    p_y = p_y / ((b2 - b3) - (((a2 - a3) * (b1 - b3)) / (a1 - a3)))
    p_x = ((a2 * (a1 - a3)) + (b2 * (b1 - b3)) - ((b1 - b3) * p_y)) / (a1 - a3)
    principal_point = (p_x, p_y)
    print("principal point: ", principal_point)

    focal_length = math.sqrt((-(p_x ** 2)) + (-(p_y ** 2)) + ((a1 + a2) * p_x) + ((b1 + b2) * p_y) - ((a1 * a2) + (b1 * b2)))
    print("focal length: ", focal_length, "\n")

    plt.title("focal length: " + str(focal_length))
    plt.imshow(cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB))
    plt.scatter([p_x], [p_y])
    plt.annotate("Principal Point", (p_x, p_y))
    plt.savefig("out/res03.jpg")

    K = np.array([[focal_length, 0, p_x],
                  [0, focal_length, p_y],
                  [0, 0, 1]
                  ])
    print("Calibration Matrix:")
    print(K, "\n")

    # Calculation Rotation Matrix
    B = (1 / (focal_length ** 2)) * np.array([
        [1, 0, -p_x],
        [0, 1, -p_y],
        [-p_x, -p_y, (focal_length ** 2) + (p_x ** 2) + (p_y ** 2)]
    ])

    v0 = np.array([
        [V_x[0]],
        [V_x[1]],
        [1]
    ])
    v1 = np.array([
        [V_y[0]],
        [V_y[1]],
        [1]
    ])
    v2 = np.array([
        [V_z[0]],
        [V_z[1]],
        [1]
    ])
    v = [v0, v1, v2]

    lambdas = []
    for i in range(3):
        lambdas.append(1 / math.sqrt(v[i].T @ B @ v[i]))

    # World-to-Camera Rotation Matrix
    R = np.linalg.inv(K) @ np.concatenate((lambdas[0] * v[0], lambdas[1] * v[1], lambdas[2] * v[2]), axis=1)
    print("Rotation Matrix R:")
    print(R, "\n")

    # World-to-Good-Camera Rotation Matrix
    R2 = np.array([
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0]
    ])

    # Camera-to-Good-Camera Rotation Matrix
    R3 = R2 @ np.linalg.inv(R)

    R3 = Rotation.from_matrix(R3).as_euler('YZX', degrees=True)
    print("R3: Y:", R3[0], "Z:", R3[1], "X:", R3[2])


if __name__ == '__main__':
    main()
