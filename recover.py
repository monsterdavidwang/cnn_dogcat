import tensorflow.keras as keras
import h5py
import gc
import itertools
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.83
# config.gpu_options.allow_growth=True
session = InteractiveSession(config=config)

def clear(a):
    del a
    gc.collect()

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
    plt.savefig(log_dir + "/fusion_matrix_new.svg", dpi=600)
    plt.show()

data_path = "/media/disk2/DogAndCat/train-data10000_200.h5"
h = h5py.File(data_path, 'r')
x = np.array(h['x'])
y = np.array(h['y'])
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=8)
clear(x)  # clear memory
clear(y)  # clear memory
clear(train_x)  # clear memory
clear(train_y)  # clear memory

log_dir = "log/log200-21"
model = keras.models.load_model(log_dir + "/epoch-1941_acc-0.883000.h5")
model.summary()

test_loss, test_acc = model.evaluate(test_x, test_y, verbose=5)
print('loss:', test_loss, ' ,accuracy:', test_acc)

pred = model.predict(test_x)
pred = np.argmax(pred, axis=1)
print(pred)
print(test_y)
conf_mat = confusion_matrix(y_true=test_y, y_pred=pred)
labels = ['cat', 'dog']
plot_confusion_matrix(conf_mat, labels)