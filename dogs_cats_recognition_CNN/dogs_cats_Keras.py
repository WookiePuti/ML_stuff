import cv2
import numpy as np
from random import shuffle
import os
from tqdm import tqdm

import tensorboard
import tensorflow as tf
import matplotlib.pyplot as plt
import time


TRAIN_DIR = '/home/lukasz/PycharmProjects/cats_dogs/train'
TEST_DIR = '/home/lukasz/PycharmProjects/cats_dogs/test'
IMG_SIZE = 50
LR = 1e-3

#MODEL_NAME = 'dogsvscatsCNN-{}'.format(int(time.time()))




def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat':
        return [1, 0]
    elif word_label == 'dog':
        return [0, 1]


def create_train_data():
    train_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        train_data.append([np.array(img), np.array(label)])
    shuffle(train_data)
    np.save('train_data.npy', train_data)
    return train_data


def proccess_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    np.save('test_data.npy', testing_data)
    return testing_data

train_data = np.load('train_data.npy', allow_pickle=True)


train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0]/255 for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array([i[1] for i in train])



sess = tf.Session()

# dense_layers = [i for i in range(3)]
# layer_sizes = [2 ** i for i in range(5, 8)]
# conv_layers = [i for i in range(1, 4)]
dense_layers = [1]
layer_sizes = [32]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            MODEL_NAME = f'{conv_layer}-conv-{layer_size}-layersize-{dense_layer}-dense-{int(time.time())}'
            tfboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(MODEL_NAME))

            model = tf.keras.Sequential()
            model.add(tf.keras. layers.Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(tf.keras.layers.Activation('relu'))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

            for l in range(conv_layer-1):
                model.add(tf.keras.layers.Conv2D(layer_size, (3, 3)))
                model.add(tf.keras.layers.Activation('relu'))
                model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

            model.add(tf.keras.layers.Flatten())

            for l in range(dense_layer):
                model.add(tf.keras.layers.Dense(layer_size))
                model.add(tf.keras.layers.Activation('relu'))

            model.add(tf.keras.layers.Dropout(0.2))


            model.add(tf.keras.layers.Dense(2))
            model.add(tf.keras.layers.Activation('sigmoid'))

            model.compile(loss='categorical_crossentropy',
                          optimizer='adam', metrics=['accuracy'])
            model.fit(X, y,  batch_size=32, validation_split=0.1, epochs=20, callbacks=[tfboard])

# test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# test_y = [i[1] for i in test]

test_data = np.load('test_data.npy', allow_pickle=True)


#model = tf.keras.models.load_model('3-conv-32-layersize-1-dense-1588071568')
fig = plt.figure()

for num, data in enumerate(test_data[-12:]):
    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    img_data = img_data/255
    #print(img_data)
    data = img_data.reshape(-1,IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]
    print(model_out)
    if np.argmax(model_out)==1:
        str_label = 'how how doge'
    else:
        str_label = 'miau mia kitku'
    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_xaxis().set_visible(False)
plt.show()











