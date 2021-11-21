import os

import cv2
import numpy as np
import tensorflow as tf


def augment(augmentation_coefficient, image_input_size, src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.10),
        tf.keras.layers.experimental.preprocessing.RandomZoom((-0.1, +0.1))
    ])

    categories_dirs = os.listdir(src_dir)
    for category_name in categories_dirs:
        os.makedirs(os.path.join(dst_dir, category_name))
        for img_file in os.listdir(os.path.join(src_dir, category_name)):
            img = cv2.imread(os.path.join(src_dir, category_name, img_file))
            img = tf.image.resize(img, image_input_size)
            img_addr = os.path.join(dst_dir, category_name, img_file).split('.')[0]
            cv2.imwrite(img_addr + '.jpg', np.array(img))
            for i in range(1, augmentation_coefficient):
                augmented_image = data_augmentation(tf.expand_dims(img, 0))
                cv2.imwrite(img_addr + '-' + str(i) + '.jpg', np.array(augmented_image[0]))


