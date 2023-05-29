import cv2
import numpy as np
import random
import h5py
import os

size = 200
train_in_path = "/media/disk2/DogAndCat/train/"
train_out_path = "/media/disk2/DogAndCat/train-200/"
train_h5_path = "/media/disk2/DogAndCat/train-data10000_200.h5"
test_in_path = "/media/disk2/DogAndCat/test/"
test_out_path = "/media/disk2/DogAndCat/test-200/"
test_h5_path = "/media/disk2/DogAndCat/test-data_200.h5"


in_path = test_in_path
out_path = test_out_path
h5_path = test_h5_path

x = np.zeros((size, size, 3))
x = np.expand_dims(x, axis=0)
y = []
count = 0

# 等比例把图像裁剪成size*size分辨率 并输出
filenames = os.listdir(in_path)
random.shuffle(filenames)
for filename in filenames:
    count += 1
    print(count)
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
        x = np.vstack((x, np.expand_dims(image/255.0, axis=0)))
        if "cat" in filename:
            s = 0
        else:
            s = 1
        y.append(s)
    else:
        n = width / float(size)
        width = size
        height = round(height / n)
        height_new = round((height - size) / 2)
        image = cv2.resize(image, (width, height))
        image = image[height_new:height_new + size, 0:size]
        x = np.vstack((x, np.expand_dims(image / 255.0, axis=0)))
        if "cat" in filename:
            s = 0
        else:
            s = 1
        y.append(s)
    cv2.imwrite(out_path + filename, image)

# save h5 file for date
x = np.delete(x, 0, 0)
y = np.array(y)
h = h5py.File(h5_path, 'w')
h['x'] = x
h['y'] = y
h.close()
