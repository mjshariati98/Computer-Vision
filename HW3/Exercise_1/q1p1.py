import numpy as np
import cv2
from skimage import feature, transform
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift


def main():
    src_image = cv2.imread('../resources/vns.jpg')
    height, width = src_image.shape[:2]

    img_copy = copyof(src_image)

    # X axis
    x_lines = x_axis_lines(img_copy)
    V_x = calculate_vanishing_point(x_lines)
    print("V_x = ", V_x)

    # Y axis
    y_lines = y_axis_lines(img_copy)
    V_y = calculate_vanishing_point(y_lines)
    print("V_y = ", V_y)

    # Z axis
    z_lines = z_axis_lines(img_copy)
    V_z = calculate_vanishing_point(z_lines)
    print("V_z = ", V_z)

    h = line_eq(V_x, V_y)
    print("h: ", "a=", h[0], ", b=", 1, ", c =", h[1], ",   a^2+b^2= ", 1 + h[0] ** 2)

    # show selected lines
    # cv2.imwrite("out/lines.jpg", img_copy)

    x_coordinates = [V_x[0], V_y[0], V_z[0]]
    y_coordinates = [V_x[1], V_y[1], V_z[1]]

    plt.imshow(cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB))
    plt.axline(V_x, V_y)
    plt.xlim([0, width])
    plt.ylim([height, 0])
    plt.savefig("out/res01.jpg")

    plt.clf()

    plt.imshow(cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB))
    plt.scatter(x_coordinates, y_coordinates)
    plt.annotate("Vx", V_x)
    plt.annotate("Vy", V_y)
    plt.annotate("Vz", V_z)
    plt.axline(V_x, V_y)
    plt.savefig("out/res02.jpg")


def copyof(image):
    copy = np.zeros(image.shape)
    copy[:, :, :] = image[:, :, :]
    return copy


# y = ax + b
def line_eq(p1, p2):
    a = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - (a * p1[0])
    return a, b


def calculate_vanishing_point(lines):
    intersects = []
    i = 0
    while i < len(lines):
        j = i + 1
        while j < len(lines):
            intersects.append(line_intersection(lines[i], lines[j]))
            j += 1
        i += 1

    points = np.array(intersects)
    mean_shift = MeanShift().fit(points)

    # calculate mean of the biggest cluster
    count = 0
    sum_x = 0
    sum_y = 0
    for i in range(len(points)):
        if mean_shift.labels_[i] == 0:  # label 0 is biggest cluster
            count += 1
            sum_x += points[i][0]
            sum_y += points[i][1]

    vp = (sum_x / count, sum_y / count)
    return vp


def line_intersection(l1, l2):
    a1, b1 = l1
    a2, b2 = l2

    x = (b2 - b1) / (a1 - a2)
    y = (a1 * x) + b1
    return x, y


# TODO read lines from file if not automate
def z_axis_lines(img_copy):
    lines = []

    p0 = (1463, 720)
    p1 = (1506, 2000)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(0, 0, 255), thickness=3)

    p0 = (1414, 720)
    p1 = (1456, 2000)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(0, 0, 255), thickness=3)

    p0 = (1135, 520)
    p1 = (1187, 2100)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(0, 0, 255), thickness=3)

    p0 = (1089, 520)
    p1 = (1120, 1500)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(0, 0, 255), thickness=3)

    p0 = (1043, 520)
    p1 = (1073, 1500)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(0, 0, 255), thickness=3)

    p0 = (997, 520)
    p1 = (1025, 1500)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(0, 0, 255), thickness=3)

    p0 = (2404, 1220)
    p1 = (2421, 1650)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(0, 0, 255), thickness=3)

    p0 = (2450, 1320)
    p1 = (2465, 1650)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(0, 0, 255), thickness=3)

    p0 = (2485, 1220)
    p1 = (2504, 1650)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(0, 0, 255), thickness=3)

    p0 = (3481, 1250)
    p1 = (3545, 2550)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(0, 0, 255), thickness=3)

    p0 = (3419, 1250)
    p1 = (3481, 2550)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(0, 0, 255), thickness=3)

    return lines


def x_axis_lines(img_copy):
    lines = []

    p0 = (1600, 1097)
    p1 = (2000, 1175)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(0, 255, 0), thickness=3)

    p0 = (1610, 1132)
    p1 = (2010, 1209)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(0, 255, 0), thickness=3)

    p0 = (1615, 1168)
    p1 = (2015, 1240)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(0, 255, 0), thickness=3)

    p0 = (1620, 1202)
    p1 = (2020, 1275)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(0, 255, 0), thickness=3)

    p0 = (1625, 1235)
    p1 = (2025, 1306)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(0, 255, 0), thickness=3)

    p0 = (1626, 1272)
    p1 = (2025, 1341)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(0, 255, 0), thickness=3)

    # p0 = (3620, 1440)
    # p1 = (4010, 1518)
    # line = line_eq(p0, p1)
    # x_lines.append(line)
    # cv2.line(img_copy, p0, p1, color=(0, 255, 0), thickness=3)
    #
    # p0 = (3620, 1495)
    # p1 = (4010, 1570)
    # line = line_eq(p0, p1)
    # x_lines.append(line)
    # cv2.line(img_copy, p0, p1, color=(0, 255, 0), thickness=3)

    # p0 = (3620, 1532)
    # p1 = (4010, 1603)
    # line = line_eq(p0, p1)
    # x_lines.append(line)
    # cv2.line(img_copy, p0, p1, color=(0, 255, 0), thickness=3)
    #
    # p0 = (3620, 1586)
    # p1 = (4010, 1656)
    # line = line_eq(p0, p1)
    # x_lines.append(line)
    # cv2.line(img_copy, p0, p1, color=(0, 255, 0), thickness=3)

    return lines


def y_axis_lines(img_copy):
    lines = []

    p0 = (700, 699)
    p1 = (1180, 643)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(255, 0, 0), thickness=3)

    p0 = (700, 765)
    p1 = (1180, 708)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(255, 0, 0), thickness=3)

    p0 = (700, 836)
    p1 = (1180, 781)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(255, 0, 0), thickness=3)

    p0 = (700, 902)
    p1 = (1180, 848)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(255, 0, 0), thickness=3)

    p0 = (700, 968)
    p1 = (1520, 878)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(255, 0, 0), thickness=3)

    p0 = (700, 1040)
    p1 = (1520, 952)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(255, 0, 0), thickness=3)

    p0 = (700, 1111)
    p1 = (1520, 1025)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(255, 0, 0), thickness=3)

    p0 = (3050, 1427)
    p1 = (3535, 1383)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(255, 0, 0), thickness=3)

    p0 = (3050, 1482)
    p1 = (3535, 1440)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(255, 0, 0), thickness=3)

    p0 = (3050, 1536)
    p1 = (3535, 1496)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(255, 0, 0), thickness=3)

    p0 = (3050, 1573)
    p1 = (3535, 1534)
    line = line_eq(p0, p1)
    lines.append(line)
    cv2.line(img_copy, p0, p1, color=(255, 0, 0), thickness=3)

    return lines


if __name__ == '__main__':
    main()
