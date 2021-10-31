import matplotlib.pyplot as plt
import numpy as np
import os
import random
import pickle
import cv2
import random
import tensorflow as tf
from pathlib import Path
from sklearn.utils import shuffle
import tensorflow.keras
from tensorflow.keras import applications, layers, losses, optimizers, metrics
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from keras.layers.core import Lambda,Dense
import time


input = 224
path_data1 = "/home/fproenca/Tese/Data/HDA/x.pickle"
path_data2 = "/home/fproenca/Tese/Data/HDA/labels.pickle"
app = '/home/fproenca/Tese/H/junk/'
model_name = '/home/fproenca/Tese/Results/HDA/MobileNet_Class_224.h5'
call_name = '/home/fproenca/Tese/Results/HDA/triplet.h5'
path_data1_total = "/home/fproenca/Tese/Data/HDA/x_total.pickle"
path_data2_total = "/home/fproenca/Tese/Data/HDA/labels_total.pickle"
aug_val = 1
split_val = 9299
################################################################################  
def get_mobile_net(x):
	model = load_model(model_name)
	embedding = Sequential(name="Embedding")
	for layer in model.layers[:-2]: # go through until last layer
	    embedding.add(layer)
	trainable = False
	for layer in embedding.layers:
 	 if layer.name == 'mobilenet_1.00_224':
 	   for i in layer.layers:
  	    if i.name == "conv1":
  	      trainable = True
  	    i.trainable = trainable
	embedding.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
	embedding.summary()	
	feature_vectors = []
	i=0
	for xi in x:
	  feature_vectors.append(embedding.predict(np.expand_dims(xi, axis=0)))
	  i+=1	
	for j in range(len(feature_vectors)):
		feature_vectors[j] = feature_vectors[j][0]
	feature_vectors = np.array(feature_vectors)
	return feature_vectors
################################################################################  
def load_data():
  pickle_in = open(path_data1,"rb")
  x = pickle.load(pickle_in)
  x = get_mobile_net(x)
  pickle_in = open(path_data2,"rb")
  labels = pickle.load(pickle_in)
  return x,labels
################################################################################  
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal"),
  layers.experimental.preprocessing.RandomRotation(0.1),
  layers.experimental.preprocessing.RandomZoom(0.1),
  layers.experimental.preprocessing.RandomContrast(0.15),
  layers.experimental.preprocessing.RandomTranslation(0.1,0.1,fill_mode='nearest',interpolation='bilinear'),
])

def aug_data(images,labels):
  aug_img = []
  labels_aug = []
  for idxA in range(len(images)-1):
      # grab the current image and label belonging to the current
      # iteration
      currentImage = images[idxA]
      image = tf.expand_dims(currentImage, 0)
      label = labels[idxA]
      for idxB in range(aug_val):
        augmented_image = data_augmentation(image)
        aug_img.append(augmented_image[0])
        labels_aug.append(label)
  print(len(labels_aug))
  return np.array(aug_img),np.array(labels_aug)
################################################################################  
x, labels = load_data()


x_val = x[split_val:]
labels_val = labels[split_val:]
x_train = x[:split_val]
labels_train = labels[:split_val]

#aug_img,aug_label = aug_data(x_train,labels_train)
#x_train = np.concatenate((x_train, aug_img))
#labels_train = np.concatenate((labels_train, aug_label))


x_train_ =[]
x_val_ =[]

cnt = 0
for idx,image in enumerate(x_train):
  name = app + str(cnt) + 'i.npy'
  Image = np.save(name,image)
  x_train_.append(name)
  cnt +=1
for idx,image in enumerate(x_val):
  name = app + str(cnt) + 'i.npy'
  Image = np.save(name,image)
  x_val_.append(name)
  cnt +=1

x_val = x_val_
x_train = x_train_
################################################################################  
def loss1(triplet_file):

  tripletImages=[]
  triplet = [np.load(triplet_file[0]) , np.load(triplet_file[1])]
  tripletImages.append(triplet)
  tripletImages = np.array(tripletImages)
  #ap_distance = np.linalg.norm(tripletImages[0,0]  - tripletImages[0,1])
  ap_distance = tf.reduce_sum(tf.square(tripletImages[0,0]- tripletImages[0,1]), -1)

  return ap_distance.numpy()


def siamese_calc(anchor, positive, negative):
  #ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
  #print(ap_distance.numpy())
  ap_distance = np.linalg.norm(anchor  - positive)
  #print(ap_distance)
  #an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
  #print(an_distance.numpy())
  an_distance = np.linalg.norm(anchor  - negative)
  #print(an_distance)

  return (ap_distance.numpy(), an_distance.numpy())

def test_loss(ap_distance, an_distance):
  loss = ap_distance - an_distance
  #loss = max(loss + 1.0, 0.0)
  return loss


def make_triplets(images, labels,flag):
  # initialize two empty lists to hold the (image, image) pairs and
	# labels to indicate if a pair is positive or negative
  tripletImages = []
  loss_vector = []
	# calculate the total number of classes present in the dataset
	# and then build a list of indexes for each class label that
	# provides the indexes for all examples with a given label
  numClasses = len(np.unique(labels))
  print(numClasses)
  idx = [np.where(labels == i)[0] for i in range(0, numClasses+1)]
  print(idx)
  cnt = 0
  aux_cnt = 0
  stri = '.npy'
  if flag == 1:
    stri = '_.npy'
  for idxA in range(len(images)-1):
      # grab the current image and label belonging to the current
      # iteration
      currentImage = images[idxA]
      label = labels[idxA]
      for idxB in range(idxA+1,len(images)):
        loss = 0.0
        posImage = images[idxB]
        label_ = labels[idxB]
        if (label == label_):
            #print("Current:  " + str(currentImage) + " || Positive:  " + str(posImage))
            negIdx = np.where(labels != label)[0]
            ap = [currentImage, posImage]
            ap_distance = loss1(ap)
            for s in negIdx:
              negImage = images[s]
              an = [currentImage,negImage]
              an_distance = loss1(an)
              loss = test_loss(ap_distance,an_distance)
              if (loss+0.7) > 0.0 :
                tripletImage = [currentImage, posImage, negImage]
              	#print(negImage)
                name = app + str(cnt) + stri
                np.save(name,tripletImage)
                tripletImages.append(name)
                loss_vector.append(loss)
                #print("Current:  " + str(currentImage) + " || Positive:  " + str(posImage) + "|| Negative:  " + str(negImage) + "|| LOSS:  " + str(loss))            	
                cnt +=1
            aux_cnt += 1
            print("ID: " +  str(label_) + "Counter: " + str(aux_cnt))
  return (np.array(tripletImages),loss_vector)


triplet_train, train_loss  = make_triplets(x_train, labels_train,0)
triplet_val, val_loss = make_triplets(x_val, labels_val,1)

print(len(triplet_train))
print(len(triplet_val))
print(len(val_loss))
print(val_loss[2])

pickle_out = open("/home/fproenca/Tese/Data/HDA/triplet_train.pickle","wb")
pickle.dump(triplet_train, pickle_out)
pickle_out.close()

pickle_out = open("/home/fproenca/Tese/Data/HDA/triplet_val.pickle","wb")
pickle.dump(triplet_val, pickle_out)
pickle_out.close()

pickle_out = open("/home/fproenca/Tese/Data/HDA/train_loss.pickle","wb")
pickle.dump(train_loss, pickle_out)
pickle_out.close()

pickle_out = open("/home/fproenca/Tese/Data/HDA/val_loss.pickle","wb")
pickle.dump(val_loss, pickle_out)
pickle_out.close()



