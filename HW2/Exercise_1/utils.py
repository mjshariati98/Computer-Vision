import cv2
import numpy as np
import math
import sys
from scipy import signal


def save_frames(video_path, num_frames, output_dir):
    video = cv2.VideoCapture(video_path)
    success, image = video.read()
    count = 1
    while success and count <= num_frames:
        cv2.imwrite(str(output_dir) + "/frame" + str(count) + ".jpg", image)
        success, image = video.read()
        count += 1


def write_matrix_to_file(matrix, file_path):
    with open(file_path, 'wb') as f:
        for line in np.asmatrix(matrix):
            np.savetxt(f, line, fmt='%.20f')


def read_matrix_from_file(file_path):
    with open(file_path, 'r') as f:
        H = [[float(num) for num in line.split(' ')] for line in f]
    return np.array(H)


def copyof(image):
    copy = np.zeros(image.shape)
    copy[:, :, :] = image[:, :, :]
    return copy


def crop(image, i, j, PIECE_SIZE, layer):
    crop = np.zeros(image[i:i + PIECE_SIZE, j:j + PIECE_SIZE, layer].shape)
    crop[:, :] = image[i:i + PIECE_SIZE, j:j + PIECE_SIZE, layer]
    return crop


def find_homography_RANSAC(src, dst):
    # RGB -> GRAY
    im1_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

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
        if m.distance / n.distance < 0.90:
            final_match_points.append(m)

    # Find Homography
    # Get Src(im2) and Dst(im1) match points for cv2.findHomography function
    src_points = []
    dst_points = []
    for match in final_match_points:
        src_points.append(interest_points_2[match.trainIdx].pt)
        dst_points.append(interest_points_1[match.queryIdx].pt)
    src_points = np.float32(src_points).reshape(-1, 1, 2)
    dst_points = np.float32(dst_points).reshape(-1, 1, 2)

    # Run RANSAC algorithm
    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 150.0)
    return H


def get_point_projection(point, H):
    project = H @ point
    project = project / project[2, 0]
    return project[0, 0], project[1, 0]


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


def create_mask(image):
    return np.vectorize(lambda x, y, z: 0 if x == 0 and y == 0 and z == 0 else 1)(image[:, :, 0], image[:, :, 1], image[:, :, 2])


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


def get_gradient(image, filter_size, sigma):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_x = get_x_derivative(image_gray, filter_size=filter_size, sigma=sigma)
    im_y = get_y_derivative(image_gray, filter_size=filter_size, sigma=sigma)

    # Calculate gradient
    return np.hypot(im_x, im_y)


def get_distributed_image(image, data_type):
    distributed_image = image / image.max() * 255
    distributed_image = np.asarray(distributed_image, dtype=data_type)
    return distributed_image


def get_thick_gradient_mask(image):
    gradient = get_gradient(image, filter_size=7, sigma=1.3)
    gradient = np.vectorize(lambda x: 255 if x > 5 else 0)(get_distributed_image(gradient, 'uint8'))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gradient = np.array(gradient, dtype='uint8')
    mask = cv2.dilate(gradient, kernel, iterations=4)
    return mask


def warp(reference, src):
    height, width = reference.shape[:2]

    # Find Homography matrix from src to ref
    H = find_homography_RANSAC(src=src, dst=reference)

    # Get minimum X and minimum Y of warped coordinates
    src_warped_min_x, src_warped_min_y, src_warped_max_x, src_warped_max_y = get_projection_of_vertices(width, height, homography=H)

    trans_x, trans_y = 0.0, 0.0
    if src_warped_min_x < 0:
        trans_x = -src_warped_min_x
    if src_warped_min_y < 0:
        trans_y = -src_warped_min_y
    T = np.array([
        [1, 0, trans_x],
        [0, 1, trans_y],
        [0, 0, 1]
    ])

    ref_warped_min_x, ref_warped_min_y, ref_warped_max_x, ref_warped_max_y = get_projection_of_vertices(width, height, T)

    # calculate width and height of panorama image
    panorama_width = int(max(src_warped_max_x + trans_x, ref_warped_max_x)) + 1
    panorama_height = int(max(src_warped_max_y + trans_y, ref_warped_max_y)) + 1
    panorama_shape = (panorama_height, panorama_width)

    # Warp src using T@H
    src_warped = cv2.warpPerspective(src, T @ H, (panorama_width, panorama_height))

    # Warp ref using T
    ref_warped = cv2.warpPerspective(reference, T, (panorama_width, panorama_height))

    src_mask = create_mask(src_warped)
    ref_mask = create_mask(ref_warped)

    return src_warped, src_mask, ref_warped, ref_mask, panorama_shape


def create_simple_panorama(src_warped, src_mask, ref_warped, ref_mask, panorama_shape):
    panorama = np.zeros((panorama_shape[0], panorama_shape[1], 3))

    for i in range(panorama_shape[1]):
        for j in range(panorama_shape[0]):
            try:
                if src_mask[j, i] == 1 and ref_mask[j, i] != 1:
                    panorama[j, i, :] = src_warped[j, i, :]
                else:
                    panorama[j, i, :] = ref_warped[j, i, :]
            except IndexError:
                pass  # left black
    return panorama


def get_laplacian_pyramid(gauss_filter, src_warped, ref_warped):
    src_warped = np.array(src_warped, dtype='float64')
    ref_warped = np.array(ref_warped, dtype='float64')

    height, width = ref_warped.shape[:2]
    src_blurred = np.zeros((height, width, 3))
    ref_blurred = np.zeros((height, width, 3))
    for layer in range(3):
        src_blurred[:, :, layer] = signal.convolve2d(src_warped[:, :, layer], gauss_filter, 'same')
        ref_blurred[:, :, layer] = signal.convolve2d(ref_warped[:, :, layer], gauss_filter, 'same')

    src_laplacian = src_warped - src_blurred
    ref_laplacian = ref_warped - ref_blurred

    src_subsamp = cv2.resize(src_blurred, None, fx=0.25, fy=0.25)
    ref_subsamp = cv2.resize(ref_blurred, None, fx=0.25, fy=0.25)

    return src_laplacian, ref_laplacian, src_subsamp, ref_subsamp


def find_overlap_area(src_mask, ref_mask):
    height, width = ref_mask.shape[:2]
    start = sys.maxsize
    end = 0

    for i in range(width):
        for j in range(height):
            if src_mask[j, i] == 1 and ref_mask[j, i] == 1:
                if i < start:
                    start = i
                elif i > end:
                    end = i

    start -= 2
    end += 2
    return start, end


def find_best_cut(src_subsamp, ref_subsamp, start_overlap, end_overlap):
    height, width = ref_subsamp.shape[:2]
    overlap = np.zeros((height, end_overlap - start_overlap))

    for layer in range(3):
        overlap += np.power(ref_subsamp[:, start_overlap:end_overlap, layer] - src_subsamp[:, start_overlap:end_overlap, layer], 2)

    return best_path(overlap)


def best_path(overlap):
    """
    Dynamic Programming to find best cut.
    :param overlap:
    :return:
    """
    E = np.zeros(overlap.shape, dtype='float64')
    h, w = overlap.shape[:2]

    for i in range(h):
        for j in range(w):
            if i == 0:
                E[i, j] = overlap[i, j]
            elif j == 0:
                E[i, j] = overlap[i, j] + min(E[i - 1, j], E[i - 1, j + 1])
            elif j == w - 1:
                E[i, j] = overlap[i, j] + min(E[i - 1, j - 1], E[i - 1, j])
            else:
                E[i, j] = overlap[i, j] + min(E[i - 1, j - 1], E[i - 1, j], E[i - 1, j + 1])

    path = []

    index = get_index(E, h - 1, -1)
    path.append(index)

    for i in range(h - 2, -1, -1):
        index = get_index(E, i, index)
        path.append(index)

    return path


def get_index(E, row, index):
    """
    used in `get_best_path` function to find best cut with DP.
    :param E:
    :param row:
    :param index:
    :return:
    """
    if index == -1:
        minimum = min(E[row])
        for i in range(len(E[row])):
            if E[row, i] == minimum:
                index = i
    else:
        if index == 0:
            minimum = min(E[row, index:index + 2])
            for i in range(index, index + 2):
                if E[row, i] == minimum:
                    index = i
        elif index == E.shape[1] - 1:
            minimum = min(E[row, index - 1:index + 1])
            for i in range(index - 1, index + 1):
                if E[row, i] == minimum:
                    index = i
        else:
            minimum = min(E[row, index - 1:index + 2])
            for i in range(index - 1, index + 2):
                if E[row, i] == minimum:
                    index = i

    return index


def fill_panorama(src_warped, ref_warped, src_mask, ref_mask, panorama=None, prev_start_overlap=None, prev_end_overlap=None, prev_best_cut=None):
    panorama_height, panorama_width = ref_warped.shape[:2]

    gauss_filter = gaussian_filter(size=13, sigma=4.0)
    src_laplacian, ref_laplacian, src_subsamp, ref_subsamp = get_laplacian_pyramid(gauss_filter, src_warped, ref_warped)

    start_overlap, end_overlap = find_overlap_area(src_mask, ref_mask)
    subsamp_start_overlap, subsamp_end_overlap = start_overlap // 4, end_overlap // 4
    subsamp_best_cut = find_best_cut(src_subsamp, ref_subsamp, subsamp_start_overlap, subsamp_end_overlap)
    best_cut = []
    for i in range(len(subsamp_best_cut)):
        best_cut.append(subsamp_best_cut[i] * 4)

    # Layer 2 of Laplacian Pyramid: Subsamp panorama
    subsamp_panorama = np.zeros((panorama_height // 4, panorama_width // 4, 3))
    subsamp_src_mask = cv2.resize(np.array(src_mask, dtype='float64'), None, fx=0.25, fy=0.25)
    subsamp_ref_mask = cv2.resize(np.array(ref_mask, dtype='float64'), None, fx=0.25, fy=0.25)

    # Fill subsamp_panorama (non-overlap area)
    for i in range(panorama_width // 4):
        for j in range(panorama_height // 4):
            try:
                if subsamp_src_mask[j, i] == 1 and subsamp_ref_mask[j, i] != 1:
                    subsamp_panorama[j, i, :] = src_subsamp[j, i, :]
                elif subsamp_ref_mask[j, i] == 1 and subsamp_src_mask[j, i] != 1:
                    subsamp_panorama[j, i, :] = ref_subsamp[j, i, :]
            except IndexError:
                pass
    # Fill subsamp_panorama (overlap area)
    h, w = ref_subsamp.shape[0], subsamp_end_overlap - subsamp_start_overlap
    for i in range(h):
        for j in range(w):
            try:
                if j <= subsamp_best_cut[h - i - 1]:
                    subsamp_panorama[i, subsamp_start_overlap + j] = src_subsamp[i, subsamp_start_overlap + j, :]
                else:
                    subsamp_panorama[i, subsamp_start_overlap + j] = ref_subsamp[i, subsamp_start_overlap + j, :]
            except IndexError:
                pass

    blurred_panorama = cv2.resize(subsamp_panorama, None, fx=4, fy=4)

    # Layer 1 of Laplacian Pyramid: Laplacian
    laplacian = np.zeros((panorama_height, panorama_width, 3))
    # Fill laplacian (non-overlap area)
    for i in range(blurred_panorama.shape[1]):
        for j in range(blurred_panorama.shape[0]):
            try:
                if src_mask[j, i] == 1 and ref_mask[j, i] != 1:
                    laplacian[j, i, :] = src_laplacian[j, i, :]
                elif ref_mask[j, i] == 1 and src_mask[j, i] != 1:
                    laplacian[j, i, :] = ref_laplacian[j, i, :]
            except IndexError:
                pass
    # Fill laplacian (overlap area)
    h, w = ref_warped.shape[0], (end_overlap - start_overlap)
    for i in range(h):
        for j in range(w):
            try:
                if j <= best_cut[(h // 4) - (i // 4) - 1]:
                    laplacian[i, start_overlap + j] = src_laplacian[i, start_overlap + j, :]
                else:
                    laplacian[i, start_overlap + j] = ref_laplacian[i, start_overlap + j, :]
            except IndexError:
                pass

    # Build new panorama
    new_panorama = blurred_panorama + laplacian

    if panorama is not None:
        h, w = panorama_height, prev_end_overlap - prev_start_overlap
        for i in range(panorama_height):
            for j in range(panorama_width):
                try:
                    if j > prev_best_cut[(h // 4) - (i // 4) - 1]:
                        panorama[i, prev_start_overlap + j] = new_panorama[i, prev_start_overlap + j, :]
                except IndexError:
                    pass
    else:
        panorama = new_panorama

    return panorama, start_overlap, end_overlap, best_cut


def update(panorama, warped_pic, mask1, mask2):
    for i in range(warped_pic.shape[0]):
        for j in range(warped_pic.shape[1]):
            if mask1[i, j] == 1 and mask2[i, j] == 1:
                warped_pic[i, j, :] = panorama[i, j, :]
    return warped_pic


def get_valid_points(good_points, prev_good_points, valid_points):
    valid_points_indices = []
    for i in range(len(valid_points)):
        if valid_points[i] == 1:
            valid_points_indices.append(i)
    return good_points[valid_points_indices], prev_good_points[valid_points_indices]


def smooth_trajectory(transforms, window_size):
    f = np.ones(window_size) / window_size
    transforms = np.lib.pad(transforms, (window_size // 2, window_size // 2), mode='edge')
    return np.convolve(transforms, f, mode='same')[window_size // 2:-window_size // 2]
