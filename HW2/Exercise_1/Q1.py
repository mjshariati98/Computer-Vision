import cv2
import numpy as np
import os
import gc
import math
from statistics import median
from utils import save_frames, copyof, find_homography_RANSAC, get_point_projection, get_projection_of_vertices, create_mask, warp, \
    create_simple_panorama, write_matrix_to_file, read_matrix_from_file, fill_panorama, update, crop, get_thick_gradient_mask, get_valid_points, \
    smooth_trajectory

FRAME_NUMBERS = 900
FPS = 30


def main():
    # Read video and save it frame by frame in a temporary folder
    # if not os.cut.exists('tmp'):
    #     os.mkdir('tmp')
    #     os.mkdir('tmp/homo')
    #     os.mkdir('tmp/warp')
    #     os.mkdir('tmp/background')
    #     os.mkdir('tmp/foreground')
    #     os.mkdir('tmp/wide')

    # save_frames('../resources/video.mp4', FRAME_NUMBERS, 'tmp')

    part1()
    # part2()
    # part3()
    # part4()
    # part5()
    # part6()
    # part7()
    # part8()

    # Remove temporary folder
    # os.rmdir('tmp')


def part1():
    frame270 = cv2.imread('tmp/frame270.jpg')
    frame450 = cv2.imread('tmp/frame450.jpg')

    # Find Homography matrix from 270 to 450
    H = find_homography_RANSAC(src=frame270, dst=frame450)
    H_inverse = np.linalg.inv(H)

    # Draw rectangle on frame450
    p1, p2 = (400, 300), (1200, 1000)
    rect450 = cv2.rectangle(copyof(frame450), p1, p2, (0, 0, 255), 2)
    cv2.imwrite('out/res01-450-rect.jpg', rect450)

    # Calculate projection of above rectangle and draw it on frame270
    p1 = get_point_projection(np.array([[400], [300], [1]]), H_inverse)
    p2 = get_point_projection(np.array([[1200], [300], [1]]), H_inverse)
    p3 = get_point_projection(np.array([[1200], [1000], [1]]), H_inverse)
    p4 = get_point_projection(np.array([[400], [1000], [1]]), H_inverse)

    rect270 = copyof(frame270)
    cv2.line(rect270, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color=(0, 0, 255), thickness=2)
    cv2.line(rect270, (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1])), color=(0, 0, 255), thickness=2)
    cv2.line(rect270, (int(p3[0]), int(p3[1])), (int(p4[0]), int(p4[1])), color=(0, 0, 255), thickness=2)
    cv2.line(rect270, (int(p4[0]), int(p4[1])), (int(p1[0]), int(p1[1])), color=(0, 0, 255), thickness=2)
    cv2.imwrite('out/res02-270-rect.jpg', rect270)

    src_warped, src_mask, ref_warped, ref_mask, panorama_shape = warp(reference=frame450, src=frame270)
    panorama = create_simple_panorama(src_warped, src_mask, ref_warped, ref_mask, panorama_shape)
    cv2.imwrite('out/res03-270-450-panorama.jpg', panorama)


def part2():
    frame90 = cv2.imread('tmp/frame90.jpg')
    frame270 = cv2.imread('tmp/frame270.jpg')
    frame450 = cv2.imread('tmp/frame450.jpg')
    frame630 = cv2.imread('tmp/frame630.jpg')
    frame810 = cv2.imread('tmp/frame810.jpg')

    height, width = frame450.shape[:2]

    # H_270 = find_homography_RANSAC(src=frame270, dst=frame450)
    # H_630 = find_homography_RANSAC(src=frame630, dst=frame450)
    # H_90_270 = find_homography_RANSAC(src=frame90, dst=frame270)
    # H_90 = H_90_270 @ H_270
    # H_810_630 = find_homography_RANSAC(src=frame810, dst=frame630)
    # H_810 = H_810_630 @ H_630
    #
    # write_matrix_to_file(H_90, 'frame90')
    # write_matrix_to_file(H_270, 'frame270')
    # write_matrix_to_file(H_630, 'frame630')
    # write_matrix_to_file(H_810, 'frame810')

    H_90 = read_matrix_from_file('tmp/homo/frame90')
    H_270 = read_matrix_from_file('tmp/homo/frame270')
    H_630 = read_matrix_from_file('tmp/homo/frame630')
    H_810 = read_matrix_from_file('tmp/homo/frame810')

    # Calculate width and height of panorama
    min_x_90, min_y_90, max_x_90, max_y_90 = get_projection_of_vertices(width, height, homography=H_90)
    min_x_270, min_y_270, max_x_270, max_y_270 = get_projection_of_vertices(width, height, homography=H_270)
    min_x_630, min_y_630, max_x_630, max_y_630 = get_projection_of_vertices(width, height, homography=H_630)
    min_x_810, min_y_810, max_x_810, max_y_810 = get_projection_of_vertices(width, height, homography=H_810)

    min_x = min(min_x_90, min_x_270, min_x_630, min_x_810)
    min_y = min(min_y_90, min_y_270, min_y_630, min_y_810)

    panorama_width = int(max(max_x_90, max_x_270, max_x_630, max_x_810))
    if min_x < 0:
        panorama_width += -int(min_x)
    if panorama_width % 4 != 0:
        panorama_width += (4 - (panorama_width % 4))
    panorama_height = int(max(max_y_90, max_y_270, max_y_630, max_y_810))
    if min_y < 0:
        panorama_height += -int(min_y)
    if panorama_height % 4 != 0:
        panorama_height = panorama_height + (4 - (panorama_height % 4))

    T = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ])

    warped_90 = cv2.warpPerspective(frame90, T @ H_90, (panorama_width, panorama_height))
    warped_270 = cv2.warpPerspective(frame270, T @ H_270, (panorama_width, panorama_height))
    warped_450 = cv2.warpPerspective(frame450, T, (panorama_width, panorama_height))
    warped_630 = cv2.warpPerspective(frame630, T @ H_630, (panorama_width, panorama_height))
    warped_810 = cv2.warpPerspective(frame810, T @ H_810, (panorama_width, panorama_height))

    mask_90 = create_mask(warped_90)
    mask_270 = create_mask(warped_270)
    mask_450 = create_mask(warped_450)
    mask_630 = create_mask(warped_630)
    mask_810 = create_mask(warped_810)

    # Fill panorama
    # Frame 90 and 270
    panorama, start_overlap, end_overlap, best_cut = fill_panorama(warped_90, warped_270, mask_90, mask_270)
    # Add Frame 450
    warped_270 = update(panorama, warped_270, mask_90, mask_270)
    panorama, start_overlap, end_overlap, best_cut = fill_panorama(warped_270, warped_450, mask_270, mask_450,
                                                                   panorama, start_overlap, end_overlap, best_cut)
    # Add Frame 630
    warped_450 = update(panorama, warped_450, mask_270, mask_450)
    panorama, start_overlap, end_overlap, best_cut = fill_panorama(warped_450, warped_630, mask_450, mask_630,
                                                                   panorama, start_overlap, end_overlap, best_cut)
    # Add Frame 810
    warped_630 = update(panorama, warped_630, mask_450, mask_630)
    panorama, start_overlap, end_overlap, best_cut = fill_panorama(warped_630, warped_810, mask_630, mask_810,
                                                                   panorama, start_overlap, end_overlap, best_cut)
    cv2.imwrite('out/res04-key-frames-panorama.jpg', panorama)


def part3():
    frame450 = cv2.imread('tmp/frame450.jpg')
    frame270 = cv2.imread('tmp/frame270.jpg')
    frame630 = cv2.imread('tmp/frame630.jpg')

    height, width = frame450.shape[:2]

    # H_270 = find_homography_RANSAC(src=frame270, dst=frame450)
    # H_630 = find_homography_RANSAC(src=frame630, dst=frame450)
    #
    # frame_num = 1
    # while frame_num <= FRAME_NUMBERS:
    #     frame = cv2.imread('tmp/frame' + str(frame_num) + '.jpg')
    #     if frame_num < 270:
    #         H = find_homography_RANSAC(src=frame, dst=frame270)
    #         H = H @ H_270
    #     elif frame_num > 630:
    #         H = find_homography_RANSAC(src=frame, dst=frame630)
    #         H = H @ H_630
    #     else:
    #         H = find_homography_RANSAC(src=frame, dst=frame450)
    #
    #     write_matrix_to_file(H, 'tmp/homo/frame' + str(frame_num))
    #     frame_num += 1

    # Calculate width and height of panorama
    minimum_x, minimum_y, maximum_x, maximum_y = 0, 0, 0, 0
    frame_num = 1
    while frame_num <= 900:
        H = read_matrix_from_file('tmp/homo/frame' + str(frame_num))
        min_x, min_y, max_x, max_y = get_projection_of_vertices(width, height, homography=H)
        if min_x < minimum_x:
            minimum_x = min_x
        if min_y < minimum_y:
            minimum_y = min_y
        if max_x > maximum_x:
            maximum_x = max_x
        if max_y > maximum_y:
            maximum_y = max_y
        frame_num += 1

    panorama_width = int(maximum_x)
    if minimum_x < 0:
        panorama_width += -int(minimum_x)
    panorama_height = int(maximum_y)
    if minimum_y < 0:
        panorama_height += -int(minimum_y)

    T = np.array([
        [1, 0, -minimum_x],
        [0, 1, -minimum_y],
        [0, 0, 1]
    ])
    write_matrix_to_file(T, 'tmp/homo/T')

    # frame_num = 1
    # while frame_num <= 900:
    #     frame = cv2.imread('tmp/frame' + str(frame_num) + '.jpg')
    #     H = read_matrix_from_file('tmp/homo/frame' + str(frame_num))
    #     warped = cv2.warpPerspective(frame, T @ H, (panorama_width, panorama_height))
    #     cv2.imwrite('tmp/warp/frame' + str(frame_num) + '.jpg', warped)
    #     frame_num += 1

    video = cv2.VideoWriter('out/res05-reference-plane.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (panorama_width, panorama_height))
    frame_num = 1
    while frame_num <= FRAME_NUMBERS:
        warped_frame = cv2.imread('tmp/warp/frame' + str(frame_num) + '.jpg')
        video.write(warped_frame)
        frame_num += 1
    video.release()


def part4():
    frame_450 = cv2.imread('tmp/warp/frame450.jpg')
    height, width = frame_450.shape[:2]

    result = np.zeros((height, width, 3))

    PIECE_SIZE = 700  # almost 10GB memory
    for i in range(0, height, PIECE_SIZE):
        for j in range(0, width, PIECE_SIZE):
            print(i, j)
            frames_l0 = []
            frames_l1 = []
            frames_l2 = []
            frame_num = 1
            while frame_num <= FRAME_NUMBERS:
                # print(frame_num)
                warped_frame = cv2.imread('tmp/warp/frame' + str(frame_num) + '.jpg')
                frames_l0.append(crop(warped_frame, i, j, PIECE_SIZE, 0))
                frames_l1.append(crop(warped_frame, i, j, PIECE_SIZE, 1))
                frames_l2.append(crop(warped_frame, i, j, PIECE_SIZE, 2))
                frame_num += 1

            median_l0 = np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]), 0, frames_l0)
            median_l1 = np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]), 0, frames_l1)
            median_l2 = np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]), 0, frames_l2)

            result[i:i + PIECE_SIZE, j:j + PIECE_SIZE, 0] = median_l0
            result[i:i + PIECE_SIZE, j:j + PIECE_SIZE, 1] = median_l1
            result[i:i + PIECE_SIZE, j:j + PIECE_SIZE, 2] = median_l2

            cv2.imwrite('out/res06-background-panorama.jpg', result)


def part5():
    main_height, main_width = cv2.imread('tmp/frame1.jpg').shape[:2]
    background_panorama = cv2.imread('out/‫‪res06-background-panorama.jpg')

    T = read_matrix_from_file('tmp/homo/T')

    frames = []
    frame_num = 1
    while frame_num <= FRAME_NUMBERS:
        H_frame = read_matrix_from_file('tmp/homo/frame' + str(frame_num))
        H = T @ H_frame
        H_inv = np.linalg.inv(H)
        warped = cv2.warpPerspective(background_panorama, H_inv, (main_width, main_height))
        frames.append(warped)
        cv2.imwrite('tmp/background/frame' + str(frame_num) + '.jpg', warped)
        frame_num += 1

    video = cv2.VideoWriter('out/res07-background-video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (main_width, main_height))
    for frame in frames:
        video.write(frame)
    video.release()


def part6():
    main_height, main_width = cv2.imread('tmp/frame1.jpg').shape[:2]
    THRESHOLD = 7000
    frames = []

    frame_num = 1
    while frame_num <= FRAME_NUMBERS:
        frame = np.array(cv2.imread('tmp/frame' + str(frame_num) + '.jpg'), dtype='float64')
        background_frame = cv2.imread('tmp/background/frame' + str(frame_num) + '.jpg')
        background_mask = get_thick_gradient_mask(background_frame)

        def is_foreground(frame_l0, frame_l1, frame_l2, background_frame_l0, background_frame_l1, background_frame_l2, background_mask):
            if background_mask != 255 and math.fabs(frame_l0 - background_frame_l0) ** 2 + math.fabs(frame_l1 - background_frame_l1) ** 2 + \
                    math.fabs(frame_l2 - background_frame_l2) ** 2 > THRESHOLD:
                return frame_l0, frame_l1, 255
            return frame_l0, frame_l1, frame_l2

        foreground_layers = np.vectorize(is_foreground)(frame[:, :, 0], frame[:, :, 1], frame[:, :, 2], background_frame[:, :, 0],
                                                        background_frame[:, :, 1], background_frame[:, :, 2], background_mask)
        foreground_frame = np.zeros(background_frame.shape)
        for layer in range(3):
            foreground_frame[:, :, layer] = foreground_layers[layer]

        foreground_frame = np.array(foreground_frame, dtype='uint8')
        frames.append(foreground_frame)
        cv2.imwrite('tmp/foreground/foreground' + str(frame_num) + '.jpg', foreground_frame)
        frame_num += 1

    video = cv2.VideoWriter('out/res08-foreground-video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (main_width, main_height))
    for frame in frames:
        video.write(frame)
    video.release()


def part7():
    WIDER_WIDTH = 3000  # initial width is 1920 -> increase 1080 pixels, 56%

    main_height, main_width = cv2.imread('tmp/frame1.jpg').shape[:2]
    background_panorama = cv2.imread('out/res06-background-panorama.jpg‬‬')
    panorama_height, panorama_width = background_panorama.shape[:2]

    T = read_matrix_from_file('tmp/homo/T')

    frames = []
    frame_num = 1
    while frame_num <= FRAME_NUMBERS:
        H_frame = read_matrix_from_file('tmp/homo/frame' + str(frame_num))
        H = T @ H_frame
        H_inv = np.linalg.inv(H)

        min_x, min_y, max_x, max_y = get_projection_of_vertices(panorama_width, panorama_height, homography=H_inv)
        if frame_num > FRAME_NUMBERS / 2 and max_x < WIDER_WIDTH:
            break

        warped = cv2.warpPerspective(background_panorama, H_inv, (WIDER_WIDTH, main_height))
        frames.append(warped)
        cv2.imwrite('tmp/wide/frame' + str(frame_num) + '.jpg', warped)
        frame_num += 1

    video = cv2.VideoWriter('out/res09-background-video-wider.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (WIDER_WIDTH, main_height))
    for frame in frames:
        video.write(frame)
    video.release()


def part8():
    SMOOTH_WINDOW = 60
    height, width = cv2.imread('tmp/frame1.jpg').shape[:2]

    x_transforms = []
    y_transforms = []
    angle_transforms = []

    prev_frame = None
    prev_frame_good_points = None
    frame_num = 1
    while frame_num <= FRAME_NUMBERS:
        frame = cv2.imread('tmp/frame' + str(frame_num) + '.jpg')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is None:  # frame 1
            prev_frame = gray_frame
            prev_frame_good_points = cv2.goodFeaturesToTrack(gray_frame, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
            frame_num += 1
            continue

        good_points, valid_points, _ = cv2.calcOpticalFlowPyrLK(prev_frame, gray_frame, prev_frame_good_points, None)
        # Choose only valid points
        good_points, prev_frame_good_points = get_valid_points(good_points, prev_frame_good_points, valid_points)

        # Find transformation matrix between 2 frames and separate delta_x, delta_y and delta_angle
        transform_matrix = cv2.estimateAffinePartial2D(prev_frame_good_points, good_points)[0]
        delta_x = transform_matrix[0, 2]
        delta_y = transform_matrix[1, 2]
        delta_angle = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])

        x_transforms.append(delta_x)
        y_transforms.append(delta_y)
        angle_transforms.append(delta_angle)

        prev_frame = gray_frame
        prev_frame_good_points = cv2.goodFeaturesToTrack(gray_frame, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        frame_num += 1

    # calculate trajectory
    trajectory_x = np.cumsum(x_transforms)
    trajectory_y = np.cumsum(y_transforms)
    trajectory_angle = np.cumsum(angle_transforms)

    # Smooth trajectory
    smoothed_trajectory_x = smooth_trajectory(trajectory_x, window_size=SMOOTH_WINDOW)
    smoothed_trajectory_y = smooth_trajectory(trajectory_y, window_size=SMOOTH_WINDOW)
    smoothed_trajectory_angle = smooth_trajectory(trajectory_angle, window_size=SMOOTH_WINDOW)

    # Smooth transformations
    smooth_x_transforms = x_transforms + (smoothed_trajectory_x - trajectory_x)
    smooth_y_transforms = y_transforms + (smoothed_trajectory_y - trajectory_y)
    smooth_angle_transforms = angle_transforms + (smoothed_trajectory_angle - trajectory_angle)

    frames = []
    frame_num = 1
    while frame_num < FRAME_NUMBERS:
        frame = cv2.imread('tmp/frame' + str(frame_num) + '.jpg')

        delta_x = smooth_x_transforms[frame_num - 1]
        delta_y = smooth_y_transforms[frame_num - 1]
        delta_angle = smooth_angle_transforms[frame_num - 1]

        affine_matrix = np.array([
            [np.cos(delta_angle), -np.sin(delta_angle), delta_x],
            [np.sin(delta_angle), np.cos(delta_angle), delta_y]
        ])

        warped = cv2.warpAffine(frame, affine_matrix, (width, height))[15:-15, 15:-15, :]
        frames.append(warped)

        frame_num += 1

    video = cv2.VideoWriter('out/res10-video-shakeless.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (width - 30, height - 30))
    for frame in frames:
        video.write(frame)
    video.release()


if __name__ == '__main__':
    main()
