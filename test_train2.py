import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
import numpy as np
import h5py
import time
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.83
# config.gpu_options.allow_growth=True
session=InteractiveSession(config=config)


def CNN(train_x, train_y, test_x, test_y):
    log_dir = "log/log100-1"
    tf_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=6, kernel_size=5, strides=1),
        keras.layers.MaxPool2D(pool_size=5, strides=2),
        keras.layers.Conv2D(filters=20, kernel_size=5, strides=2),
        keras.layers.MaxPool2D(pool_size=5, strides=2),
        keras.layers.Flatten(),
        keras.layers.Dense(units=1024, activation='relu'),
        keras.layers.Dropout(rate=0.8, seed=8),
        keras.layers.Dense(units=256, activation='sigmoid'),
        keras.layers.Dense(units=2, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(0.005),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )

    model.fit(train_x, train_y, batch_size=128, epochs=500, validation_data=(test_x, test_y), callbacks=[tf_callback])

    test_loss, test_acc = model.evaluate(test_x, test_y, verbose=5)
    print('loss:', test_loss, ' ,accuracy:', test_acc)


if __name__ == '__main__':
    data_path = "/media/disk2/DogAndCat/train-data_100.h5"
    h = h5py.File(data_path, 'r')
    x = np.array(h['x'])
    y = np.array(h['y'])
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=8)
    print(train_x.shape, test_x.shape)
    stime = time.time()
    CNN(train_x, train_y, test_x, test_y)
    etime = time.time()
    print("training time:", etime-stime, "seconds.")

# Terminal: tensorboard --logdir=logs
