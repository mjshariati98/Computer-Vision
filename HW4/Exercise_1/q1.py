import os
import shutil
import random
import multiprocessing

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import roc_curve, plot_precision_recall_curve, average_precision_score
import torch
import torchvision
import matplotlib.patches as patches

SRC_POSITIVE_DIR = '../resources/lfw'
SRC_NEGATIVE_DIR = '../resources/256_ObjectCategories'
SHUFFLED_POSITIVE_DIR = '../resources/data/positive'
SHUFFLED_NEGATIVE_DIR = '../resources/data/negative'

TRAIN_NUM = 10000
VALID_NUM = 1000
TEST_NUM = 1000

# hyperparameters
BASE_SIZE = (64, 64)
NUMBER_OF_BINS = 9
PIXELS_PER_CELL = (6, 6)
CELLS_PER_BLOCK = (3, 3)
BLOCK_NORM = 'L2-Hys'


def main():
    # shuffle data
    shuffle_data(SRC_POSITIVE_DIR, SHUFFLED_POSITIVE_DIR, TRAIN_NUM, VALID_NUM, TEST_NUM)
    shuffle_data(SRC_NEGATIVE_DIR, SHUFFLED_NEGATIVE_DIR, TRAIN_NUM, VALID_NUM, TEST_NUM)

    # calculate hog feature vectors of each images
    positive_data = {}
    negative_data = {}

    positive_data['train'] = calculate_feature_vectors(os.path.join(SHUFFLED_POSITIVE_DIR, 'train'))
    # write_vectors_to_file(positive_data['train'], "p_train_6.6_3.3")
    positive_data['validation'] = calculate_feature_vectors(os.path.join(SHUFFLED_POSITIVE_DIR, 'validation'))
    # write_vectors_to_file(positive_data['validation'], "p_valid_6.6_3.3")
    positive_data['test'] = calculate_feature_vectors(os.path.join(SHUFFLED_POSITIVE_DIR, 'test'))
    # write_vectors_to_file(positive_data['test'], "p_test_6.6_3.3")

    negative_data['train'] = calculate_feature_vectors(os.path.join(SHUFFLED_NEGATIVE_DIR, 'train'))
    # write_vectors_to_file(negative_data['train'], "n_train_6.6_3.3")
    negative_data['validation'] = calculate_feature_vectors(os.path.join(SHUFFLED_NEGATIVE_DIR, 'validation'))
    # write_vectors_to_file(negative_data['validation'], "n_valid_6.6_3.3")
    negative_data['test'] = calculate_feature_vectors(os.path.join(SHUFFLED_NEGATIVE_DIR, 'test'))
    # write_vectors_to_file(negative_data['test'], "n_test_6.6_3.3")

    # positive_data['train'] = read_vectors_from_file("p_train_6.6_3.3")
    # positive_data['validation'] = read_vectors_from_file("p_valid_6.6_3.3")
    # positive_data['test'] = read_vectors_from_file("p_test_6.6_3.3")

    # negative_data['train'] = read_vectors_from_file("n_train_6.6_3.3")
    # negative_data['validation'] = read_vectors_from_file("n_valid_6.6_3.3")
    # negative_data['test'] = read_vectors_from_file("n_test_6.6_3.3")

    # Train
    train_data = positive_data['train']
    train_data.extend(negative_data['train'])
    train_labels = [1 for i in range(TRAIN_NUM)]
    train_labels.extend([-1 for i in range(TRAIN_NUM)])

    svm_classifier = svm.SVC(kernel='rbf', cache_size=2000)
    svm_classifier.fit(train_data, train_labels)

    # Validation
    positive_predict = svm_classifier.predict(positive_data['validation'])
    negative_predict = svm_classifier.predict(negative_data['validation'])

    correct = 0
    for res in positive_predict:
        if res == 1:
            correct += 1
    for res in negative_predict:
        if res == -1:
            correct += 1

    print("Validation Accuracy is: ", (correct / (2 * VALID_NUM)) * 100, "%")

    # Test
    positive_predict = svm_classifier.predict(positive_data['test'])
    negative_predict = svm_classifier.predict(negative_data['test'])

    correct = 0
    for res in positive_predict:
        if res == 1:
            correct += 1
    for res in negative_predict:
        if res == -1:
            correct += 1

    print("Test Accuracy is: ", (correct / (2 * TEST_NUM)) * 100, "%")

    # plot ROC and PrecessionRecall curves
    test_data = positive_data['test']
    test_data.extend(negative_data['test'])
    test_labels = [1 for i in range(TEST_NUM)]
    test_labels.extend([-1 for i in range(TEST_NUM)])
    test_labels = np.array(test_labels)
    test_score = svm_classifier.decision_function(test_data)

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(test_labels, test_score)
    # plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='orange', label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig('out/res1.jpg')

    # compute AP
    average_precision = average_precision_score(test_labels, test_score)
    print('AveragePrecision(AP) is', average_precision)

    # plot PrecessionRecall curve
    plt.clf()
    precision_recall_curve = plot_precision_recall_curve(svm_classifier, test_data, test_labels)
    precision_recall_curve.ax_.set_title('AP={0:0.5f}'.format(average_precision))
    plt.savefig('out/res2.jpg')

    # run Face Detector on samples
    melli = cv2.imread("../resources/Melli.jpg")
    face_detector(melli, svm_classifier, threshold=1.5, scales=[1])
    plt.savefig("out/res4.jpg")

    persepolis = cv2.imread("../resources/Persepolis.jpg")
    face_detector(persepolis, svm_classifier, scales=[1.0, 1.2])
    plt.savefig("out/res5.jpg")

    esteghlal = cv2.imread("../resources/Esteghlal.jpg")
    face_detector(esteghlal, svm_classifier, threshold=0.7, scales=[1.5, 1.75])
    plt.savefig("out/res6.jpg")


def face_detector(img, classifier, threshold=1.0, base_wind_size=64, scales=None, trans=1, iou_threshold=0.2):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if scales is None:
        scales = [0.0625, 0.125, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    params = []
    for scale in scales:
        wind_size = int(base_wind_size * scale)
        params.append((gray_img, classifier, wind_size, trans, threshold))

    with multiprocessing.Pool(processes=4) as pool:
        results = pool.starmap(scaled_face_detector, params)

    boxes = []
    scores = []
    for box, score in results:
        boxes.extend(box)
        scores.extend(score)

    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)
    final_BBs = torchvision.ops.nms(boxes=boxes, scores=scores, iou_threshold=iou_threshold)

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for idx in final_BBs:
        rect = patches.Rectangle((float(boxes[idx][0]), float(boxes[idx][1])), float(boxes[idx][2]) - float(boxes[idx][0]),
                                 float(boxes[idx][3]) - float(boxes[idx][1]), edgecolor='green', facecolor='none', linewidth=2)
        ax.add_patch(rect)
        ax.text(float(boxes[idx][0]), float(boxes[idx][1]) - 5, 'Score:{:.2f}'.format(scores[idx]), color='green')


def scaled_face_detector(gray_img, classifier, wind_size, trans, threshold):
    boxes = []
    scores = []
    height, width = gray_img.shape
    for i in range(0, width, trans):
        print(i)
        for j in range(0, height, trans):
            window = gray_img[j:j + wind_size, i:i + wind_size]
            if window.shape != (wind_size, wind_size):
                continue
            resized_window = cv2.resize(window, BASE_SIZE)
            fv = hog(resized_window, orientations=NUMBER_OF_BINS, pixels_per_cell=PIXELS_PER_CELL,
                     cells_per_block=CELLS_PER_BLOCK, block_norm=BLOCK_NORM)
            score = classifier.decision_function([fv])
            if score[0] > threshold:
                boxes.append([i, j, i + wind_size, j + wind_size])
                scores.append(score[0])
    return boxes, scores


def shuffle_data(dir, shuffled_dir, train_num, valid_num, test_num):
    all_images_paths = []
    for folder in os.listdir(dir):
        for img_file in os.listdir(os.path.join(dir, folder)):
            all_images_paths.append(os.path.join(dir, folder, img_file))

    random_sample = random.sample(range(0, len(all_images_paths)), train_num + valid_num + test_num)

    train_data = [all_images_paths[i] for i in random_sample[:train_num]]
    valid_data = [all_images_paths[i] for i in random_sample[train_num:train_num + valid_num]]
    test_data = [all_images_paths[i] for i in random_sample[train_num + valid_num:]]

    mv_shuffled_data(shuffled_dir, train_data, valid_data, test_data)


def mv_shuffled_data(shuffled_dir, train_data, valid_data, test_data):
    train_path = os.path.join(shuffled_dir, 'train')
    valid_path = os.path.join(shuffled_dir, 'validation')
    test_path = os.path.join(shuffled_dir, 'test')
    try:
        os.makedirs(train_path)
        os.makedirs(valid_path)
        os.makedirs(test_path)
    except OSError:
        print("shuffled directory was exists")
        return

    for dst_path, images in [(train_path, train_data), (valid_path, valid_data), (test_path, test_data)]:
        for img in images:
            shutil.copy(img, dst_path)


def calculate_feature_vectors(images_path):
    feature_vectors = []
    for img_file in os.listdir(images_path):
        img = cv2.cvtColor(cv2.imread(os.path.join(images_path, img_file)), cv2.COLOR_RGB2GRAY)
        resized_img = cv2.resize(img, BASE_SIZE)

        # calculate feature vector using hog function from skimage library
        fv = hog(resized_img, orientations=NUMBER_OF_BINS, pixels_per_cell=PIXELS_PER_CELL,
                 cells_per_block=CELLS_PER_BLOCK, block_norm=BLOCK_NORM)
        feature_vectors.append(fv)
    return feature_vectors


def write_vectors_to_file(vectors, file_path):
    with open(file_path, 'wb') as f:
        for line in vectors:
            np.savetxt(f, line, fmt='%.20f', newline=' ')
            f.write(b'\n')


def read_vectors_from_file(file_path):
    vectors = []
    with open(file_path, 'r') as f:
        for line in f:
            vec = []
            for num in line.split(' '):
                if num != '\n':
                    vec.append(float(num))
            vectors.append(np.array(vec))
    return vectors


if __name__ == '__main__':
    main()
