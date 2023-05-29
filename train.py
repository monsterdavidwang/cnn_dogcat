import itertools
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
import numpy as np
import h5py
import time
import gc
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.83
# config.gpu_options.allow_growth=True
session = InteractiveSession(config=config)


def CNN(train_x, train_y, test_x, test_y):
    tf_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model_name = '/epoch-{epoch:03d}_acc-{val_sparse_categorical_accuracy:03f}.h5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=log_dir + model_name,
        monitor='val_sparse_categorical_accuracy',
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        mode='max',
        period=1)

    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=3, kernel_size=11, strides=1),
        keras.layers.MaxPool2D(pool_size=5, strides=2),
        keras.layers.Dropout(rate=0.5, seed=8),
        keras.layers.Conv2D(filters=10, kernel_size=9, strides=2),
        keras.layers.MaxPool2D(pool_size=5, strides=2),
        keras.layers.Dropout(rate=0.5, seed=8),
        keras.layers.Conv2D(filters=20, kernel_size=7, strides=2),
        keras.layers.MaxPool2D(pool_size=5, strides=2),
        keras.layers.Dropout(rate=0.5, seed=8),
        keras.layers.Flatten(),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dropout(rate=0.4, seed=8),
        keras.layers.Dense(units=128, activation='tanh'),
        keras.layers.Dense(units=2, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(0.005),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )

    model.fit(train_x, train_y, batch_size=64, epochs=2000, validation_data=(test_x, test_y), callbacks=[tf_callback, checkpoint])
    clear(train_x)  # clear memory
    clear(train_y)  # clear memory

    test_loss, test_acc = model.evaluate(test_x, test_y, verbose=5)
    print('loss:', test_loss, ' ,accuracy:', test_acc)

    pred = model.predict(test_x)
    pred = np.argmax(pred, axis=1)
    print(pred)
    print(test_y)
    conf_mat = confusion_matrix(y_true=test_y, y_pred=pred)
    labels = ['cat', 'dog']
    plot_confusion_matrix(conf_mat, labels)


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylim(len(cm) - 0.5, -0.5)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(log_dir + "/fusion_matrix.svg", dpi=600)
    plt.show()


def clear(a):
    del a
    gc.collect()

if __name__ == '__main__':
    log_dir = "log/log200-23"
    data_path = "/media/disk2/DogAndCat/train-data10000_200.h5"
    h = h5py.File(data_path, 'r')
    x = np.array(h['x'])
    y = np.array(h['y'])
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=8)
    clear(x)  # clear memory
    clear(y)  # clear memory
    print(train_x.shape, test_x.shape)
    stime = time.time()
    CNN(train_x, train_y, test_x, test_y)
    etime = time.time()
    print("training time:", etime-stime, "seconds.")

# Terminal: tensorboard --logdir=logs
