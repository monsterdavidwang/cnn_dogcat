import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
import cv2
import numpy as np
import random
import h5py
import os
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
            n = height / float(size)
            height = size
            width = round(width / n)
            width_new = round((width - size) / 2)
            image = cv2.resize(image, (width, height))
            image = image[0:size, width_new:width_new + size]
        else:
            n = width / float(size)
            width = size
            height = round(height / n)
            height_new = round((height - size) / 2)
            image = cv2.resize(image, (width, height))
            image = image[height_new:height_new + size, 0:size]
        cv2.imwrite(out_path + filename, image)

def sort(size, path):
    filenames = os.listdir(path)
    random.shuffle(filenames)
    x = np.zeros((size, size, 3))
    x = np.expand_dims(x, axis=0)
    y = []
    count = 0
    for filename in filenames[0:20000]:
        count += 1
        train_in = os.path.join(path, filename)
        image = cv2.imread(train_in)
        image = image / 255.0
        if "cat" in filename:
            s = 0
        else:
            s = 1
        x = np.vstack((x, np.expand_dims(image, axis=0)))
        y.append(s)
        print(count)
    x = np.delete(x, 0, 0)
    y = np.array(y)
    print(111)
    print(222)
    return x, y


if __name__ == '__main__':
    size = 100
    train_in_path = "/media/disk2/DogAndCat/train/"
    train_out_path = "/media/disk2/DogAndCat/train-cut/"
    test_in_path = "/media/disk2/DogAndCat/test/"
    test_out_path = "/media/disk2/DogAndCat/test-cut/"
    # data_cut(size, train_in_path, train_out_path)
    # data_cut(size, test_in_path, test_out_path)
    # train_x, train_y, test_x, test_y = sort(size, train_out_path)
    x,y = sort(size, train_out_path)
    print(x.shape,y.shape)
    # print(train_x.shape, train_y.shape)
    # print(test_x.shape, test_y.shape)
    h = h5py.File('/media/disk2/DogAndCat/train-cut/data.h5','w')
    h['x'] = x
    h['y'] = y
    # h['train_x'] = train_x2
    # h['train_y'] = train_y
    # h['test_x'] = test_x
    # h['test_y'] = test_y
    h.close()
    # CNN(train_x,train_y,test_x,test_y)
