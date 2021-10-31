import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.preprocessing import image
import random
import tensorflow.keras

def load_data():

      filenames = os.listdir('/home/fproenca/Tese/Datasets/CUHK01')
      x = []
      labels = []

      j=0
      for filename in filenames:
        x += [cv2.imread(os.path.join(os.path.abspath('/home/fproenca/Tese/Datasets/CUHK01'), filename), cv2.IMREAD_UNCHANGED)]
        labels += [int(filename[:4])]
        j +=1
        print(j)

      for i in range(len(x)):
        x[i] = cv2.resize(x[i], (224,224), interpolation=cv2.INTER_LINEAR).astype('uint8')
        #x[i]= cv2.copyMakeBorder(x[i], 0, 0, 32, 32, cv2.BORDER_CONSTANT)
        tensorflow.keras.applications.mobilenet.preprocess_input(x[i])
        print(i)

      return x, labels

x, labels = load_data()

idx = np.argsort(labels)
x = np.array(x)[idx]
labels = np.array(labels)[idx]

pickle_out = open("/home/fproenca/Tese/Data/CUHK01/x_total.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("/home/fproenca/Tese/Data/CUHK01/labels_total.pickle","wb")
pickle.dump(labels, pickle_out)
pickle_out.close()

pickle_in = open("/home/fproenca/Tese/Data/CUHK01/x_total.pickle","rb")
x_total = pickle.load(pickle_in)

pickle_in = open("/home/fproenca/Tese/Data/CUHK01/labels_total.pickle","rb")
labels_total = pickle.load(pickle_in)

#Separate between train and test
x = x_total[:3484]
labels = labels_total[:3484]

pickle_out = open("/home/fproenca/Tese/Data/CUHK01/x.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("/home/fproenca/Tese/Data/CUHK01/labels.pickle","wb")
pickle.dump(labels, pickle_out)
pickle_out.close()

pickle_in = open("/home/fproenca/Tese/Data/CUHK01/x.pickle","rb")
x = pickle.load(pickle_in)

pickle_in = open("/home/fproenca/Tese/Data/CUHK01/labels.pickle","rb")
labels = pickle.load(pickle_in)
