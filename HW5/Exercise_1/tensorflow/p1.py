import os
import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from utils import normalize_image, plot_loss_and_accuracy, is_data_augmented
from data_augmentation import augment

TRAIN_AUGMENTED_DIR = 'augmented_data/train'
VALID_AUGMENTED_DIR = 'augmented_data/valid'
CLASS_NUM = 15
IMAGE_INPUT_SIZE = (227, 227)
BATCH_SIZE = 32


def main():
    tf.config.threading.set_inter_op_parallelism_threads(6)
    tf.config.threading.set_intra_op_parallelism_threads(6)

    if not is_data_augmented():
        augment()

    # load images as dataset
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_AUGMENTED_DIR,
        seed=123,
        image_size=IMAGE_INPUT_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True)

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        VALID_AUGMENTED_DIR,
        seed=123,
        image_size=IMAGE_INPUT_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True)

    # normalize images
    train_dataset = train_dataset.map(normalize_image)
    validation_dataset = validation_dataset.map(normalize_image)

    # define model
    alexnet_model = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4)),

        keras.layers.Flatten(),

        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(CLASS_NUM, activation='softmax')
    ])

    # compile model
    alexnet_model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.optimizers.SGD(learning_rate=0.001),
                          metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)])
    alexnet_model.summary()

    # train the network
    alexnet_model.fit(train_dataset,
                      epochs=50,
                      validation_data=validation_dataset,
                      validation_freq=1)

    # plot loss, top-1 accuracy and top-5 accuracy
    plot_loss_and_accuracy(alexnet_model)
    plt.savefig("out/1.jpg")


if __name__ == '__main__':
    main()
