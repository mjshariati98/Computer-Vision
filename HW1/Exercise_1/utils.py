import sys
import cv2
import math
import numpy as np
from scipy import signal


# Gaussian function
def gaussian(x, y, mu, sigma):
    return (1 / (2 * math.pi * sigma ** 2)) * math.exp(-(((x - mu) ** 2 + (y - mu) ** 2) / (2 * sigma ** 2)))


# Derivative of Gaussian function
def gaussian_derivative(x, y, mu, sigma):
    return ((-(x - mu)) / (2 * math.pi * sigma ** 4)) * math.exp(-(((x - mu) ** 2 + (y - mu) ** 2) / (2 * sigma ** 2)))


def gaussian_filter(size, sigma):
    """
    build a size*size filter of gaussian function and return it.

    :param size: size of filter(matrix)
    :param sigma: the sigma of gaussian function
    :return: filter
    """
    mu = size // 2
    mat = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            mat[i, j] = gaussian(j, i, mu, sigma)

    return mat


def gaussian_derivative_filter(size, sigma, axis):
    """
    build a size*size filter of derivative of gaussian function and return it.

    :param size: size of filter(matrix)
    :param sigma: the sigma of gaussian function
    :param axis: the x-coordinate or y-coordinate
    :return: filter
    """
    mu = size // 2
    mat = np.zeros((size, size))
    if axis == 'x':
        for i in range(size):
            for j in range(size):
                mat[i, j] = gaussian_derivative(j, i, mu, sigma)
    else:
        for i in range(size):
            for j in range(size):
                mat[i, j] = gaussian_derivative(i, j, mu, sigma)

    return mat


def get_x_derivative(image, filter_size, sigma):
    return signal.convolve2d(image, gaussian_derivative_filter(filter_size, sigma, 'x'), 'same')


def get_y_derivative(image, filter_size, sigma):
    return signal.convolve2d(image, gaussian_derivative_filter(filter_size, sigma, 'y'), 'same')


def get_distributed_image(image, data_type):
    distributed_image = image / image.max() * 255
    distributed_image = np.asarray(distributed_image, dtype=data_type)
    return distributed_image


def calculate_Ix2_Iy2_Ixy(I_x, I_y):
    I_x2 = np.multiply(I_x, I_x)
    I_y2 = np.multiply(I_y, I_y)
    I_xy = np.multiply(I_x, I_y)
    return I_x2, I_y2, I_xy


def calculate_Sx2_Sy2_Sxy(I_x2, I_y2, I_xy, gaussian_filter_size, gaussian_filter_sigma):
    S_x2 = signal.convolve2d(I_x2, gaussian_filter(gaussian_filter_size, gaussian_filter_sigma), 'same')
    S_y2 = signal.convolve2d(I_y2, gaussian_filter(gaussian_filter_size, gaussian_filter_sigma), 'same')
    S_xy = signal.convolve2d(I_xy, gaussian_filter(gaussian_filter_size, gaussian_filter_sigma), 'same')
    return S_x2, S_y2, S_xy


def calculate_R(height, width, S_x2, S_y2, S_xy, k):
    R = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            det_M = (S_x2[y, x] * S_y2[y, x]) - (S_xy[y, x] ** 2)
            trace_M = S_x2[y, x] + S_y2[y, x]
            R[y, x] = det_M - k * (trace_M ** 2)
    return R


def apply_threshold(image, threshold):
    return np.vectorize(lambda x: x if x > threshold else 0, otypes=[float])(
        get_distributed_image(image, data_type='float32'))


# Non-Maximum Suppression
def nms(image):
    height, width = image.shape[0], image.shape[1]

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        image=np.asarray(image, dtype='uint8'), connectivity=8, ltype=cv2.CV_32S)

    mapping = {}
    for j in range(height):
        for i in range(width):
            try:
                if image[j, i] > mapping[labels[j, i]][1]:
                    mapping[labels[j, i]] = ((j, i), image[j, i])
            except KeyError:
                mapping[labels[j, i]] = ((j, i), image[j, i])

    points = []
    for label in mapping:
        points.append(mapping[label][0])

    return points


def show_points_on_image(image, points):
    new_image = np.zeros(image.shape)
    new_image[:, :, :] = image

    for point in points:
        # interest point
        new_image[point[0], point[1]] = [0, 0, 255]
        # also around interest point to be more clear
        new_image[point[0] - 1, point[1]] = [0, 0, 255]
        new_image[point[0] + 1, point[1]] = [0, 0, 255]
        new_image[point[0], point[1] - 1] = [0, 0, 255]
        new_image[point[0], point[1] + 1] = [0, 0, 255]
        new_image[point[0] - 1, point[1] - 1] = [0, 0, 255]
        new_image[point[0] + 1, point[1] + 1] = [0, 0, 255]
        new_image[point[0] - 1, point[1] + 1] = [0, 0, 255]
        new_image[point[0] + 1, point[1] - 1] = [0, 0, 255]

    return new_image


def get_match_points(image1, image2, points1, points2, n, ratio):
    height, width = image1.shape[0], image1.shape[1]

    match_points = {}
    more_than_one_corresponds = []
    for point1 in points1:
        first_match = (sys.maxsize, (0, 0))
        second_match = (sys.maxsize, (0, 0))
        window1 = get_sub_window(image1, point1, n)
        if window1 is None:
            continue
        for point2 in points2:
            if math.fabs(point1[0] - point2[0]) > height / 5 or math.fabs(point1[1] - point2[1]) > width / 5:
                continue

            window2 = get_sub_window(image2, point2, n)
            if window2 is None:
                continue

            window1 = np.array(window1, dtype='float64')
            window2 = np.array(window2, dtype='float64')

            sum = 0
            for layer in range(3):
                diff = np.power(window1[:, :, layer] - window2[:, :, layer], 2)
                sum += np.sum(diff)

            if sum <= first_match[0]:
                second_match = first_match
                first_match = (sum, point2)
            elif sum <= second_match[0]:
                second_match = (sum, point2)

        d1, d2 = first_match[0], second_match[0]
        if d1 / d2 < ratio:
            if first_match[1] not in match_points.values():
                match_points[point1] = first_match[1]

    # if one point matches to more than one point, all of matches must be deleted.
    for val in more_than_one_corresponds:
        for key in match_points:
            if match_points[key] == val:
                match_points.pop(key)
                break

    return match_points


def get_sub_window(image, point, n):
    y, x = point
    height, width = image.shape[0], image.shape[1]
    if y - n // 2 >= 0 and x - n // 2 >= 0 and y + n // 2 <= height and x + n // 2 <= width:
        if len(image.shape) == 3:
            return image[y - n // 2:y + n // 2, x - n // 2:x + n // 2, :]
        else:
            return image[y - n // 2:y + n // 2, x - n // 2:x + n // 2]
    return None


def get_random_color():
    color = (list(np.random.choice(range(256), size=3)))
    return [int(color[0]), int(color[1]), int(color[2])]
