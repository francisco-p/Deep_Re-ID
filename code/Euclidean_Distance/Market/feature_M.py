import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Activation, BatchNormalization, Flatten, InputLayer, Input, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.preprocessing import image
import random
from sklearn.utils import shuffle, class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, average_precision_score, confusion_matrix, precision_recall_curve, auc
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from itertools import combinations
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

pickle_in = open("/home/fproenca/Tese/Data/Market/x.pickle","rb")
x = pickle.load(pickle_in)

pickle_in = open("/home/fproenca/Tese/Data/Market/labels.pickle","rb")
labels = pickle.load(pickle_in)

#############################################################################

def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        if input_img.ndim == 3:
            img_h, img_w, img_c = input_img.shape
        elif input_img.ndim == 2:
            img_h, img_w = input_img.shape

        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            if input_img.ndim == 3:
                c = np.random.uniform(v_l, v_h, (h, w, img_c))
            if input_img.ndim == 2:
                c = np.random.uniform(v_l, v_h, (h, w))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w] = c

        return input_img

    return eraser

aug = ImageDataGenerator(
  preprocessing_function=get_random_eraser(v_l=0, v_h=1),
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

#############################################################################
input = 224

def base_model():

    mobilenet = tensorflow.keras.applications.mobilenet.MobileNet(input_shape=(input ,input ,3), weights="imagenet", include_top=False)
    #mobilenet = tensorflow.keras.applications.MobileNetV2(input_shape=(input ,input ,3), weights="imagenet", include_top=False)

    model = Sequential()
    model.add(mobilenet)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(751, activation='softmax'))

    return model


def train_valid_split(x,labels):
  
  x_classes = []
  labels_classes = []

  for j in range(1, 751):
    aux_x = []
    aux_labels = []
    for i in range(len(labels)):
      if labels[i] == j:
        aux_x.append(x[i])
        aux_labels.append(labels[i])
    x_classes.append(aux_x)
    labels_classes.append(aux_labels)
  
  x_train = []
  labels_train = []
  x_valid = []
  labels_valid = []

  for i in range(len(x_classes)):
    aux_x_t, aux_x_v, aux_labels_t, aux_labels_v = train_test_split(x_classes[i], labels_classes[i], test_size=0.33,random_state = 42)

    for a in range(len(aux_x_t)):
      x_train.append(aux_x_t[a])
      labels_train.append(aux_labels_t[a])

    for b in range(len(aux_x_v)):
      x_valid.append(aux_x_v[b])
      labels_valid.append(aux_labels_v[b])

  x_train = np.array(x_train)
  labels_train = np.array(labels_train)
  x_valid = np.array(x_valid)
  labels_valid = np.array(labels_valid)

  return x_train, labels_train, x_valid, labels_valid


x_train, labels_train, x_valid, labels_valid = train_valid_split(x,labels)

aux = np.arange(labels_train.shape[0])
np.random.seed(2)
np.random.shuffle(aux)
x_train = x_train[aux]
labels_train = labels_train[aux]

aux = np.arange(labels_valid.shape[0])
np.random.seed(3)
np.random.shuffle(aux)
x_valid = x_valid[aux]
labels_valid = labels_valid[aux]

labels_train = to_categorical(labels_train, num_classes=751)
labels_valid = to_categorical(labels_valid, num_classes=751)

model = base_model()
model.summary()

lr_schedule = tensorflow.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)

opt = tensorflow.keras.optimizers.SGD(learning_rate=lr_schedule)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics = ['accuracy'])

callbacks = ModelCheckpoint('/home/fproenca/Tese/Results/Market/MobileNet_Class_224.h5', monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=False, mode='auto')



results = model.fit(x=aug.flow(x_train, labels_train,batch_size=16), epochs=300, callbacks = callbacks, validation_data=(x_valid, labels_valid))
def display_training_curves(training, validation, title, subplot):
  if subplot%10==1: # set up the subplots on the first call
    plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
    plt.tight_layout()
  ax = plt.subplot(subplot)
  ax.set_facecolor('#F8F8F8')
  ax.plot(training)
  ax.plot(validation)
  ax.set_title('model '+ title)
  ax.set_ylabel(title)
  ax.set_xlabel('epoch')
  ax.legend(['train', 'valid.'])


print(results.history.keys())
display_training_curves(results.history['accuracy'], results.history['val_accuracy'], 'accuracy', 211)
display_training_curves(results.history['loss'], results.history['val_loss'], 'loss', 212)

model = load_model('/home/fproenca/Tese/Results/Market/MobileNet_Class_224')

predictions = []

for i in range(len(x_valid)):
  if i%1000 == 0:
    print(i)
  predictions.append(model.predict(np.expand_dims(x_valid[i], axis=0))[0])

labels_valid = labels_valid.tolist()

for i in range(len(predictions)):
  predictions[i] = np.argmax(predictions[i])
  labels_valid[i] = np.array(labels_valid[i])
  labels_valid[i] = np.argmax(labels_valid[i])

predictions = np.array(predictions)
labels_valid = np.array(labels_valid)

print('Accuracy: %.3f' % accuracy_score(labels_valid, predictions))
print('Precision: %.3f' % precision_score(labels_valid, predictions, average='micro'))
print('Recall: %.3f' % recall_score(labels_valid, predictions, average='micro'))
#print('AP: %.3f' % average_precision_score(labels_valid, predictions))
coo = confusion_matrix(labels_valid, predictions)
print(classification_report(labels_valid, predictions))




