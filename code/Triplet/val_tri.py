import matplotlib.pyplot as plt
import numpy as np
import os
import random
import pickle
import cv2
import tensorflow as tf
from pathlib import Path
from sklearn.utils import shuffle
import tensorflow.keras
from tensorflow.keras import applications, layers, losses, optimizers, metrics
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from keras.layers.core import Lambda,Dense


input = 224
triplet_train_name = '/home/fproenca/Tese/Data/HDA/triplet_train.pickle'
triplet_val_name = '/home/fproenca/Tese/Data/HDA/triplet_val.pickle'
train_loss = "/home/fproenca/Tese/Data/HDA/train_loss.pickle"
val_loss = "/home/fproenca/Tese/Data/HDA/val_loss.pickle"

margin = 0.5
##################################################################################


pickle_in = open(triplet_train_name,"rb")
triplet_train = pickle.load(pickle_in)

pickle_in = open(triplet_val_name,"rb")
triplet_val = pickle.load(pickle_in)

pickle_in = open(train_loss,"rb")
train_loss = pickle.load(pickle_in)

pickle_in = open(val_loss,"rb")
val_loss = pickle.load(pickle_in)



def prepare_triplets(triplet,loss):
 tri_final = []
 for idx,tri in enumerate(triplet):
  l = loss[idx]
  if l+margin > 0.0:
   tri_final.append(tri)
 return(np.array(tri_final))
  
triplet_train = prepare_triplets(triplet_train,train_loss)
triplet_val = prepare_triplets(triplet_val,val_loss)


print(len(triplet_train))
print(len(triplet_val))

a = len(triplet_train) + len(triplet_val)

print(" N Triplets: " + str(a))
