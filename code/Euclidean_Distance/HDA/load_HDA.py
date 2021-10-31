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

def load_data():

      filenames = os.listdir('/home/fproenca/Tese/Datasets/HDA+')

      x = []
      labels = []
      labels_new = []

      j=0
      for filename in filenames:
        x += [cv2.imread(os.path.join(os.path.abspath('/home/fproenca/Tese/Datasets/HDA+'), filename), cv2.IMREAD_UNCHANGED)]
        pos = str.find(filename,'_')
        labels += [int(filename[:pos])]
        j +=1
        if j%500 == 0:
          print(j)
      
      for i in range(len(x)):
        x[i] = cv2.resize(x[i], (224,224), interpolation=cv2.INTER_LINEAR).astype('uint8')
        #x[i]= cv2.copyMakeBorder(x[i], 0, 0, 32, 32, cv2.BORDER_CONSTANT)
        tensorflow.keras.applications.mobilenet.preprocess_input(x[i])

      idx = np.argsort(labels)
      x = np.array(x)[idx]
      labels = np.array(labels)[idx]

      counter = 0
      for idx,number  in enumerate(labels):
        #print(" labels[idx]: " + str( labels[idx]) + " || labels[idx-1]: " + str(labels[idx-1]) + " || counter: " + str(counter))
        if labels[idx] != labels[idx-1]:
          counter += 1
        labels_new += [counter]
        np.array(labels_new)

      return  x , labels_new


x, labels = load_data()

pickle_out = open("/home/fproenca/Tese/Data/HDA/x_total.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("/home/fproenca/Tese/Data/HDA/labels_total.pickle","wb")
pickle.dump(labels, pickle_out)
pickle_out.close()

pickle_in = open("/home/fproenca/Tese/Data/HDA/x_total.pickle","rb")
x_total = pickle.load(pickle_in)

pickle_in = open("/home/fproenca/Tese/Data/HDA/labels_total.pickle","rb")
labels_total = pickle.load(pickle_in)

x = x_total[:10988]
labels = labels_total[:10988]


pickle_out = open("/home/fproenca/Tese/Data/HDA/x.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("/home/fproenca/Tese/Data/HDA/labels.pickle","wb")
pickle.dump(labels, pickle_out)
pickle_out.close()

pickle_in = open("/home/fproenca/Tese/Data/HDA/x.pickle","rb")
x = pickle.load(pickle_in)

pickle_in = open("/home/fproenca/Tese/Data/HDA/labels.pickle","rb")
labels = pickle.load(pickle_in)




















