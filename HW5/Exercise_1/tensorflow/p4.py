import os
import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import torchvision

from utils import load_data, normalize_image, plot_loss_and_accuracy, is_data_augmented
from data_augmentation import augment

TRAIN_AUGMENTED_DIR = 'augmented_data/train'
VALID_AUGMENTED_DIR = 'augmented_data/valid'
CLASS_NUM = 15
IMAGE_INPUT_SIZE = (227, 227)


def main():
    # tf.config.threading.set_inter_op_parallelism_threads(6)
    # tf.config.threading.set_intra_op_parallelism_threads(6)

    if not is_data_augmented():
        augment()

    # train_images, train_labels = load_data(TRAIN_AUGMENTED_DIR, IMAGE_INPUT_SIZE)
    # validation_images, validation_labels = load_data(VALID_AUGMENTED_DIR, IMAGE_INPUT_SIZE)
    #
    # train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    # validation_dataset = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))
    #
    # train_ds_size = tf.data.experimental.cardinality(train_dataset).numpy()
    # validation_ds_size = tf.data.experimental.cardinality(validation_dataset).numpy()
    #
    # train_dataset = (train_dataset
    #                  .map(normalize_image)
    #                  .shuffle(buffer_size=train_ds_size)
    #                  .batch(batch_size=32, drop_remainder=False))
    # validation_dataset = (validation_dataset
    #                       .map(normalize_image)
    #                       .shuffle(buffer_size=validation_ds_size)
    #                       .batch(batch_size=32, drop_remainder=False))

    base_model = tf.keras.applications.MobileNetV2(input_shape=IMAGE_INPUT_SIZE + (3,),
                                                   include_top=False,
                                                   weights='imagenet')
    # Freeze base model layers and train only the last one
    base_model.trainable = False
    base_model.summary()

    input_layer = tf.keras.Input(shape=IMAGE_INPUT_SIZE + (3,))
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    dropout = tf.keras.layers.Dropout(0.2)
    output_layer = tf.keras.layers.Dense(units=CLASS_NUM)

    # MobileNetV2 expects pixel value of images in range [-1, 1]. So using preprocess_input of mobilenet_v2 to rescale them.
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    model = tf.keras.Model(
        input_layer,
        output_layer(
            dropout(
                global_average_layer(
                    base_model(
                        preprocess_input(input_layer),
                        training=False
                    )
                )
            )
        )
    )

    model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.optimizers.SGD(lr=0.0001),
                  metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)])
    model.summary()

    model.fit(train_dataset,
              epochs=50,
              validation_data=validation_dataset,
              validation_freq=1)

    # plot loss, top-1 accuracy and top-5 accuracy
    plot_loss_and_accuracy(model)
    plt.savefig("out/4.jpg")


if __name__ == '__main__':
    main()
