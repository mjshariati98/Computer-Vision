import numpy as np
import cv2


def show_together(image1, image2):
    height1, width1 = image1.shape[0], image1.shape[1]
    height2, width2 = image2.shape[0], image2.shape[1]

    im_dst = np.zeros((height1, width1 + width2, 3))
    im_dst[:, :width1, :] = image1
    im_dst[:height2, width1:, :] = image2
    return im_dst


def copyof(image):
    copy = np.zeros(image.shape)
    copy[:, :, :] = image[:, :, :]
    return copy


def draw_lines_on_together_image(image1, image2, point1, point2):
    together = show_together(image1, image2)
    point2 = (point2[0] + image1.shape[1], point2[1])
    together = cv2.line(together, point1, point2, (0, 0, 255), thickness=3)

    return together
