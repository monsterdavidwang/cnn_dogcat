import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
import cv2
import numpy as npp
import mars.tensor as np
import random
import os
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction=0.83
config.gpu_options.allow_growth=True
session=InteractiveSession(config=config)


# 等比例把图像裁剪成size*size分辨率 并输出
def data_cut(size, in_path, out_path):
    filenames = os.listdir(in_path)
    for filename in filenames:
        train_in = os.path.join(in_path, filename)
        image = cv2.imread(train_in)
        height = image.shape[0]
        width = image.shape[1]
        if (height <= width):
            n = height / 100.0
            height = 100
            width = round(width / n)
            width_new = round((width - 100) / 2)
            image = cv2.resize(image, (width, height))
            image = image[0:100, width_new:width_new + 100]
        else:
            n = width / 100.0
            width = 100
            height = round(height / n)
            height_new = round((height - 100) / 2)
            image = cv2.resize(image, (width, height))
            image = image[height_new:height_new + 100, 0:100]
        cv2.imwrite(out_path + filename, image)


def sort(size, path):
    filenames = os.listdir(path)
    random.seed(8)
    random.shuffle(filenames)
    x = np.zeros((size, size, 3))
    x = np.expand_dims(x, axis=0)
    y = []
    count = 0
    for filename in filenames[0:1000]:
        count += 1
        train_in = os.path.join(path, filename)
        image = cv2.imread(train_in) / 255.0
        if "cat" in filename:
            s = 1
        else:
            s = 0
        x = np.vstack((x, np.expand_dims(image, axis=0)))
        y.append(s)
        print(count)
    x = npp.delete(npp.array(x), 0, 0)
    y = npp.array(y)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, random_state=8)
    return train_x, train_y, test_x, test_y


def CNN(train_x, train_y, test_x, test_y):
    model = keras.models.Sequential([
        # keras.layers.Input(input_shape=(28, 28)),
        keras.layers.Conv2D(filters=6, kernel_size=5, strides=3),
        keras.layers.MaxPool2D(pool_size=2, strides=2),
        keras.layers.Conv2D(filters=18, kernel_size=2, strides=2),
        keras.layers.MaxPool2D(pool_size=2, strides=1),
        keras.layers.Flatten(),
        keras.layers.Dense(units=1024, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )

    model.fit(train_x, train_y, batch_size=32, epochs=50, validation_split=0.2)

    test_loss, test_acc = model.evaluate(test_x, test_y)
    print('loss:', test_loss, ' ,accuracy:', test_acc)


if __name__ == '__main__':
    size = 100
    train_in_path = "/media/disk2/DogAndCat/train/"
    train_out_path = "/media/disk2/DogAndCat/train-cut/"
    test_in_path = "/media/disk2/DogAndCat/test/"
    test_out_path = "/media/disk2/DogAndCat/test-cut/"
    # data_cut(size, train_in_path, train_out_path)
    # data_cut(size, test_in_path, test_out_path)
    train_x, train_y, test_x, test_y = sort(size, train_out_path)
    print(train_x.shape, train_y.shape)
    print(test_x.shape, test_y.shape)
    CNN(train_x,train_y,test_x,test_y)
