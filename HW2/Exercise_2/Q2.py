import numpy as np
import cv2


def main():
    # 3D World Points
    sample_object_points = np.zeros((6 * 9, 3), np.float32)
    sample_object_points[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    sample_object_points *= 22  # each cell is 22mm.

    # Mode-1: im01 - im10
    calibration_matrix = calculate_calibration_matrix(1, 10, sample_object_points)
    print("Mode1: im01 - im10")
    print(calibration_matrix)

    # Mode-2: im06 - im15
    calibration_matrix = calculate_calibration_matrix(6, 15, sample_object_points)
    print("Mode2: im06 - im15")
    print(calibration_matrix)

    # Mode-3: im11 - im20
    calibration_matrix = calculate_calibration_matrix(11, 20, sample_object_points)
    print("Mode3: im11 - im20")
    print(calibration_matrix)

    # Mode-4: im01 - im20
    calibration_matrix = calculate_calibration_matrix(1, 20, sample_object_points)
    print("Mode4: im01 - im20")
    print(calibration_matrix)

    # specific camera matrix
    calibration_matrix = calculate_calibration_matrix(1, 20, sample_object_points, specific=True)
    print("If principal point is in the center, skewness=0 and fx=fy :")
    print(calibration_matrix)


def calculate_calibration_matrix(from_im, to_im, sample_object_points, specific=False):
    object_points = []
    image_points = []
    height, width = 0, 0

    for i in range(from_im, to_im + 1):
        image = cv2.imread('../resources/im' + str(i) + '.jpg')
        height, width = image.shape[:2]
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Find the board corners
        ok, corners = cv2.findChessboardCorners(gray_image, (9, 6), None)
        if ok:
            object_points.append(sample_object_points)
            termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), termination_criteria)
            image_points.append(corners)

            # # Show corners
            # cv2.drawChessboardCorners(image, (9, 6), corners, ok)
            # cv2.imwrite('out/corners' + str(i) + '.jpg', image)

    if specific:
        camera_matrix = np.array([
            [1.0, 0.0, width / 2],
            [0.0, 1.0, height / 2],
            [0.0, 0.0, 1.0]
        ])
        _, camera_matrix, _, _, _ = cv2.calibrateCamera(object_points, image_points, (width, height), camera_matrix, None,
                                                        flags=cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_ASPECT_RATIO)
    else:
        _, camera_matrix, _, _, _ = cv2.calibrateCamera(object_points, image_points, (width, height), None, None)

    return camera_matrix


if __name__ == '__main__':
    main()
