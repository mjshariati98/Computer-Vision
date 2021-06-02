import numpy as np
import cv2
import os
import sys
from sklearn import neighbors

N = 14
TRAIN_ROOT_DIR = '../resources/Data/Train'
TEST_ROOT_DIR = '../resources/Data/Test'


class Category:
    def __init__(self, name, label):
        self.name = name
        self.label = label
        self.descriptors = []

    def add_train_data(self, img):
        img = np.array(img, dtype='float32')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (N, N))
        descriptor = img_resized.reshape(N * N, 1)
        self.descriptors.append(descriptor)


def main():
    # Train
    categories = []
    labels = {}
    categories_dirs = os.listdir(TRAIN_ROOT_DIR)
    for idx, category_name in enumerate(categories_dirs):
        category = Category(name=category_name, label=idx)
        labels[category_name] = idx
        for img_file in os.listdir(os.path.join(TRAIN_ROOT_DIR, category_name)):
            img = cv2.imread(os.path.join(TRAIN_ROOT_DIR, category_name, img_file))
            category.add_train_data(img)
        categories.append(category)

    # Test with NN(NearestNeighbor method)
    tests = 0
    corrects = 0
    categories_dirs = os.listdir(TEST_ROOT_DIR)
    for category_name in categories_dirs:
        category_label = labels[category_name]
        for img_file in os.listdir(os.path.join(TEST_ROOT_DIR, category_name)):
            test_img = cv2.imread(os.path.join(TEST_ROOT_DIR, category_name, img_file))
            query_label_result = nearest_neighbor(test_img, categories, norm_order=1)
            if query_label_result == category_label:
                corrects += 1
            tests += 1

    accuracy = (corrects / tests) * 100
    print("NearestNeighbor Accuracy:", accuracy)

    # Test with kNN(k-NearestNeighbor method)
    train_descriptors = []
    train_descriptors_labels = []

    for category in categories:
        for descriptor in category.descriptors:
            train_descriptors.append(descriptor.reshape(N * N))
            train_descriptors_labels.append(category.label)

    test_descriptors = []
    test_descriptors_labels = []
    categories_dirs = os.listdir(TEST_ROOT_DIR)
    for category_name in categories_dirs:
        category_label = labels[category_name]
        for img_file in os.listdir(os.path.join(TEST_ROOT_DIR, category_name)):
            test_img = np.array(cv2.imread(os.path.join(TEST_ROOT_DIR, category_name, img_file)), dtype='float32')
            test_img_resized = cv2.resize(cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY), (N, N))
            test_descriptors.append(test_img_resized.reshape(N * N))
            test_descriptors_labels.append(category_label)

    k = 5
    clf = neighbors.KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='brute', p=1)
    clf.fit(train_descriptors, train_descriptors_labels)
    query_label_result = clf.predict(test_descriptors)

    tests = 0
    corrects = 0
    for idx, query_result in enumerate(query_label_result):
        if query_result == test_descriptors_labels[idx]:
            corrects += 1
        tests += 1

    accuracy = (corrects / tests) * 100
    print("k-NearestNeighbor Accuracy:", accuracy)


def nearest_neighbor(test_img, categories, norm_order):
    test_img_resized = cv2.resize(cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY), (N, N))
    test_descriptor = test_img_resized.reshape(N * N, 1)

    min_distance = sys.maxsize
    min_label = 0
    for category in categories:
        for descriptor in category.descriptors:
            distance = np.linalg.norm((test_descriptor - descriptor), ord=norm_order)
            if distance < min_distance:
                min_distance = distance
                min_label = category.label

    return min_label


if __name__ == '__main__':
    main()
