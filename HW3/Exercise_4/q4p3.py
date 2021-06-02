import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from sklearn import neighbors, svm
from matplotlib import pyplot as plt
import seaborn


TRAIN_ROOT_DIR = '../resources/Data/Train'
TEST_ROOT_DIR = '../resources/Data/Test'


def main():
    ############################## Dictionary Learning ##############################

    descriptors = []  # all images descriptors
    categories_dirs = os.listdir(TRAIN_ROOT_DIR)
    for category_name in categories_dirs:
        for img_file in os.listdir(os.path.join(TRAIN_ROOT_DIR, category_name)):
            img = cv2.imread(os.path.join(TRAIN_ROOT_DIR, category_name, img_file))
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
            interest_points = sift.detect(img_gray, None)
            _, descriptor = sift.compute(img_gray, interest_points)
            descriptors.append(descriptor)

    descriptors = np.concatenate(descriptors, axis=0)

    # write_matrix_to_file(descriptors, "descriptors.txt")
    # descriptors = read_matrix_from_file("descriptors.txt")

    n_clusters = 50
    kmeans = KMeans(n_clusters=n_clusters).fit(descriptors)
    visual_words = kmeans.cluster_centers_

    # write_matrix_to_file(visual_words, "visual_words-" + str(n_clusters) + ".txt")
    # visual_words = read_matrix_from_file("visual_words-" + str(n_clusters) + ".txt")

    ##################################### Train #####################################

    clf_visual_words = neighbors.KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='brute', p=1)
    clf_visual_words.fit(visual_words, np.arange(0, n_clusters))

    labels = {}
    train_histograms = []
    train_histograms_labels = []
    categories_dirs = os.listdir(TRAIN_ROOT_DIR)
    for idx, category_name in enumerate(categories_dirs):
        labels[category_name] = idx
        for img_file in os.listdir(os.path.join(TRAIN_ROOT_DIR, category_name)):
            img = cv2.imread(os.path.join(TRAIN_ROOT_DIR, category_name, img_file))
            img_histogram = get_image_histogram(img, clf_visual_words, n_clusters)
            train_histograms.append(img_histogram)
            train_histograms_labels.append(labels[category_name])

    ##################################### Test #####################################

    clf_test = svm.SVC()
    clf_test.fit(train_histograms, train_histograms_labels)

    # Test with kNN(k-NearestNeighbor method)
    test_histograms = []
    test_histograms_labels = []
    categories_dirs = os.listdir(TEST_ROOT_DIR)
    for category_name in categories_dirs:
        category_label = labels[category_name]
        for img_file in os.listdir(os.path.join(TEST_ROOT_DIR, category_name)):
            test_img = cv2.imread(os.path.join(TEST_ROOT_DIR, category_name, img_file))
            test_img_histogram = get_image_histogram(test_img, clf_visual_words, n_clusters)
            test_histograms.append(test_img_histogram)
            test_histograms_labels.append(category_label)

    query_label_result = clf_test.predict(test_histograms)

    categories = len(categories_dirs)
    confusion_matrix = np.zeros((categories, categories))
    tests = 0
    corrects = 0
    for idx, query_result in enumerate(query_label_result):
        confusion_matrix[test_histograms_labels[idx], query_result] += 1
        if query_result == test_histograms_labels[idx]:
            corrects += 1
        tests += 1

    accuracy = (corrects / tests) * 100
    print("Accuracy:", accuracy, "\n")

    seaborn.heatmap(confusion_matrix, cmap="inferno", annot=confusion_matrix, annot_kws={'fontsize': 12})
    plt.savefig("out/res09.jpg")


def get_image_histogram(img, clf, n_clusters):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    interest_points = sift.detect(img_gray, None)
    _, descriptor = sift.compute(img_gray, interest_points)

    # calculate nearest visual word for every feature vector of image with kNN (k=1)
    nearest_visual_words = clf.predict(descriptor)
    img_histogram = create_histogram(n_clusters, nearest_visual_words)
    return img_histogram


def create_histogram(n_clusters, nearest_visual_words):
    histogram = [0 for i in range(n_clusters)]
    for v in nearest_visual_words:
        histogram[v] += 1

    return np.array(histogram)


def get_distributed_image(image, data_type):
    distributed_image = image / image.max() * 255
    distributed_image = np.asarray(distributed_image, dtype=data_type)
    return distributed_image


def write_matrix_to_file(matrix, file_path):
    with open(file_path, 'wb') as f:
        for line in np.asmatrix(matrix):
            np.savetxt(f, line, fmt='%.20f')


def read_matrix_from_file(file_path):
    with open(file_path, 'r') as f:
        H = [[float(num) for num in line.split(' ')] for line in f]
    return np.array(H)


if __name__ == '__main__':
    main()
