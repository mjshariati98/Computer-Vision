import cv2
import numpy as np
from utils import get_x_derivative, get_y_derivative, get_distributed_image, calculate_Ix2_Iy2_Ixy, \
    calculate_Sx2_Sy2_Sxy, calculate_R, apply_threshold, nms, show_points_on_image, get_match_points, get_random_color


def main():
    im1 = cv2.imread("../resources/im01.jpg")
    im2 = cv2.imread("../resources/im02.jpg")

    width = im1.shape[1]
    height = im1.shape[0]

    # RGB -> GRAY
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Calculate I_x and I_y: convolve images with gaussian derivative filter in x and y axises
    filter_size = 7
    sigma = 1.3
    im1_x = get_x_derivative(im1_gray, filter_size=filter_size, sigma=sigma)
    im1_y = get_y_derivative(im1_gray, filter_size=filter_size, sigma=sigma)

    im2_x = get_x_derivative(im2_gray, filter_size=filter_size, sigma=sigma)
    im2_y = get_y_derivative(im2_gray, filter_size=filter_size, sigma=sigma)

    # Calculate I_x2, I_y2 and I_xy
    im1_x2, im1_y2, im1_xy = calculate_Ix2_Iy2_Ixy(im1_x, im1_y)
    im2_x2, im2_y2, im2_xy = calculate_Ix2_Iy2_Ixy(im2_x, im2_y)

    # Calculate gradient
    im1_gradient = np.hypot(im1_x, im1_y)
    cv2.imwrite("out/res01_grad.jpg", get_distributed_image(im1_gradient, data_type='uint8'))
    im2_gradient = np.hypot(im2_x, im2_y)
    cv2.imwrite("out/res02_grad.jpg", get_distributed_image(im2_gradient, data_type='uint8'))

    # Calculate S_x2, S_y2 and S_xy
    gaussian_filter_size = 13
    gaussian_filter_sigma = 10
    im1_S_x2, im1_S_y2, im1_S_xy = calculate_Sx2_Sy2_Sxy(im1_x2, im1_y2, im1_xy,
                                                         gaussian_filter_size=gaussian_filter_size,
                                                         gaussian_filter_sigma=gaussian_filter_sigma)
    im2_S_x2, im2_S_y2, im2_S_xy = calculate_Sx2_Sy2_Sxy(im2_x2, im2_y2, im2_xy,
                                                         gaussian_filter_size=gaussian_filter_size,
                                                         gaussian_filter_sigma=gaussian_filter_sigma)

    # Build R
    R1 = calculate_R(height, width, im1_S_x2, im1_S_y2, im1_S_xy, k=0.1)
    cv2.imwrite("out/‫‪res03_score.jpg‬‬", get_distributed_image(R1, data_type='uint8'))
    R2 = calculate_R(height, width, im2_S_x2, im2_S_y2, im2_S_xy, k=0.1)
    cv2.imwrite("out/‫‪res04_score.jpg‬‬", get_distributed_image(R2, data_type='uint8'))

    # Apply threshold
    R1_threshold = apply_threshold(R1, threshold=0.5)
    cv2.imwrite("out/‫‪res05_thresh.jpg‬‬", np.vectorize(lambda x: 255 if x > 0 else 0)(R1_threshold))

    R2_threshold = apply_threshold(R2, threshold=0.5)
    cv2.imwrite("out/‫‪res06_thresh.jpg‬‬", np.vectorize(lambda x: 255 if x > 0 else 0)(R2_threshold))

    # Non-Maximum Suppression
    im1_points = nms(R1_threshold)
    im1_harris = show_points_on_image(im1, im1_points)
    cv2.imwrite("out/‫‪res07_harris.jpg‬‬", im1_harris)

    im2_points = nms(R2_threshold)
    im2_harris = show_points_on_image(im2, im2_points)
    cv2.imwrite("out/‫‪res08_harris.jpg‬‬", im2_harris)

    n = 70
    im1_match_points = get_match_points(im1, im2, im1_points, im2_points, n, ratio=0.97)
    im2_match_points = get_match_points(im2, im1, im2_points, im1_points, n, ratio=0.97)

    # Check to the corresponding points be in both matches!
    for point1 in list(im1_match_points):
        if point1 not in im2_match_points.values():
            im1_match_points.pop(point1)

    for point2 in list(im2_match_points):
        if point2 not in im1_match_points.values():
            im2_match_points.pop(point2)

    # Save result points separately
    im1_corres = show_points_on_image(im1, im1_match_points.keys())
    cv2.imwrite("out/res09_corres.jpg", im1_corres)
    im2_corres = show_points_on_image(im2, im1_match_points.values())
    cv2.imwrite("out/res10_corres.jpg", im2_corres)

    # Draw line between corresponding points
    together = np.zeros((height, width * 2, 3))
    together[:, :width] = im1
    together[:, width:] = im2
    for point1 in im1_match_points:
        point2 = (im1_match_points[point1][1] + width, im1_match_points[point1][0])
        point1 = (point1[1], point1[0])
        together = cv2.line(together, point1, point2, get_random_color(), thickness=2)
    cv2.imwrite("out/res11.jpg", together)


if __name__ == '__main__':
    main()
