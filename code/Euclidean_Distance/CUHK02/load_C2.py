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

      filenames1 = os.listdir('/home/fproenca/Tese/Datasets/CUHK02/P1/cam1/')
      filenames3 = os.listdir('/home/fproenca/Tese/Datasets/CUHK02/P2/cam1/')
      filenames5 = os.listdir('/home/fproenca/Tese/Datasets/CUHK02/P3/cam1/')
      filenames7 = os.listdir('/home/fproenca/Tese/Datasets/CUHK02/P4/cam1/')
      filenames9 = os.listdir('/home/fproenca/Tese/Datasets/CUHK02/P5/cam1/')
      filenames2 = os.listdir('/home/fproenca/Tese/Datasets/CUHK02/P1/cam2/')
      filenames4 = os.listdir('/home/fproenca/Tese/Datasets/CUHK02/P2/cam2/')
      filenames6 = os.listdir('/home/fproenca/Tese/Datasets/CUHK02/P3/cam2/')
      filenames8 = os.listdir('/home/fproenca/Tese/Datasets/CUHK02/P4/cam2/')
      filenames10 = os.listdir('/home/fproenca/Tese/Datasets/CUHK02/P5/cam2/')
      
      x = []
      labels = []
  
      for i in range(len(filenames1)):
        x += [cv2.imread(os.path.join(os.path.abspath('/home/fproenca/Tese/Datasets/CUHK02/P1/cam1/'), filenames1[i]), cv2.IMREAD_UNCHANGED)]
        labels += [int(filenames1[i][:3])]
        x += [cv2.imread(os.path.join(os.path.abspath('/home/fproenca/Tese/Datasets/CUHK02/P1/cam2/'), filenames2[i]), cv2.IMREAD_UNCHANGED)]
        labels += [int(filenames2[i][:3])]
        

      for i in range(len(labels)):
        if labels[i] > 134 and labels[i] < 628:
          labels[i] -= 1
        elif labels[i] > 628:
          labels[i] -= 2

      print('p1')

      for i in range(len(filenames3)):
        x += [cv2.imread(os.path.join(os.path.abspath('/home/fproenca/Tese/Datasets/CUHK02/P2/cam1/'), filenames3[i]), cv2.IMREAD_UNCHANGED)]
        labels += [int(filenames3[i][:3])+971]
        x += [cv2.imread(os.path.join(os.path.abspath('/home/fproenca/Tese/Datasets/CUHK02/P2/cam2/'), filenames4[i]), cv2.IMREAD_UNCHANGED)]
        labels += [int(filenames4[i][:3])+971]
       

      print('p2')
             
      for i in range(len(filenames5)):
        x += [cv2.imread(os.path.join(os.path.abspath('/home/fproenca/Tese/Datasets/CUHK02/P3/cam1/'), filenames5[i]), cv2.IMREAD_UNCHANGED)]
        labels += [int(filenames5[i][:3])+1277]
        x += [cv2.imread(os.path.join(os.path.abspath('/home/fproenca/Tese/Datasets/CUHK02/P3/cam2/'), filenames6[i]), cv2.IMREAD_UNCHANGED)]
        labels += [int(filenames6[i][:3])+1277]
     

      print('p3')

      for i in range(len(filenames7)):
        x += [cv2.imread(os.path.join(os.path.abspath('/home/fproenca/Tese/Datasets/CUHK02/P4/cam1/'), filenames7[i]), cv2.IMREAD_UNCHANGED)]
        labels += [int(filenames7[i][:3])+1385]
        x += [cv2.imread(os.path.join(os.path.abspath('/home/fproenca/Tese/Datasets/CUHK02/P4/cam2/'), filenames8[i]), cv2.IMREAD_UNCHANGED)]
        labels += [int(filenames8[i][:3])+1385]
    

      print('p4')
  
      for i in range(len(filenames9)):
        x += [cv2.imread(os.path.join(os.path.abspath('/home/fproenca/Tese/Datasets/CUHK02/P5/cam1/'), filenames9[i]), cv2.IMREAD_UNCHANGED)]
        labels += [int(filenames9[i][:3])+1578]
        x += [cv2.imread(os.path.join(os.path.abspath('/home/fproenca/Tese/Datasets/CUHK02/P5/cam2/'), filenames10[i]), cv2.IMREAD_UNCHANGED)]
        labels += [int(filenames10[i][:3])+1578]
      
      
      print('p5')
      
      for i in range(len(x)):
        x[i] = cv2.resize(x[i], (224,224), interpolation=cv2.INTER_LINEAR).astype('uint8')
        #x[i]= cv2.copyMakeBorder(x[i], 0, 0, 32, 32, cv2.BORDER_CONSTANT)
        tensorflow.keras.applications.mobilenet.preprocess_input(x[i])

      return x, labels

x, labels = load_data()

idx = np.argsort(labels)
x = np.array(x)[idx]
labels = np.array(labels)[idx]

pickle_out = open("/home/fproenca/Tese/Data/CUHK02/x_total.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("/home/fproenca/Tese/Data/CUHK02/labels_total.pickle","wb")
pickle.dump(labels, pickle_out)
pickle_out.close()

pickle_in = open("/home/fproenca/Tese/Data/CUHK02/x_total.pickle","rb")
x_total = pickle.load(pickle_in)

pickle_in = open("/home/fproenca/Tese/Data/CUHK02/labels_total.pickle","rb")
labels_total = pickle.load(pickle_in)

x = x_total[:6863]
labels = labels_total[:6863]

pickle_out = open("/home/fproenca/Tese/Data/CUHK02/x.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("/home/fproenca/Tese/Data/CUHK02/labels.pickle","wb")
pickle.dump(labels, pickle_out)
pickle_out.close()

pickle_in = open("/home/fproenca/Tese/Data/CUHK02/x.pickle","rb")
x = pickle.load(pickle_in)

pickle_in = open("/home/fproenca/Tese/Data/CUHK02/labels.pickle","rb")
labels = pickle.load(pickle_in)

