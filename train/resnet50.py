import os
from random import shuffle

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tqdm import tqdm  # Helps in vis

TEST_SIZE = 0.5
RANDOM_STATE = 2018
BATCH_SIZE = 64
NO_EPOCHS = 5
NUM_CLASSES = 2
TRAIN_DIR = '/Users/administrator/workspace/ai_data/dog_vs_cat/Kaggle/train'
TEST_DIR = '/Users/administrator/workspace/ai_data/dog_vs_cat/test'
IMG_SIZE = 224

SHORT_LIST_TRAIN = os.listdir(TRAIN_DIR)[0:10000]  # using a subset of data as resouces as limited.
SHORT_LIST_TEST = os.listdir(TEST_DIR)


def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat':
        return [1, 0]
    #                             [no cat, very doggo]
    elif word_label == 'dog':
        return [0, 1]


def create_train_data():
    training_data = []
    for img in tqdm(SHORT_LIST_TRAIN):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    return training_data


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    shuffle(testing_data)
    return testing_data


train = create_train_data()
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array([i[1] for i in train])

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='max'))
my_new_model.add(Dense(NUM_CLASSES, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = True

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

my_new_model.summary()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

train_model = my_new_model.fit(X_train, y_train,
                               batch_size=BATCH_SIZE,
                               epochs=NO_EPOCHS,
                               verbose=1,
                               validation_data=(X_val, y_val))

my_new_model.save('weights.h5')

score = my_new_model.evaluate(X_val, y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
