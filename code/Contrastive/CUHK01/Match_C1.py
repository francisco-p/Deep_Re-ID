import random
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle, class_weight
from keras.callbacks import ModelCheckpoint, TensorBoard
from model_c1 import *


epochs = 10
batch_size = 16
margin = 1  # Margin for constrastive loss.

def load_data():
  pickle_in = open("/home/fproenca/Tese/Data/CUHK01/x.pickle","rb")
  x = pickle.load(pickle_in)

  pickle_in = open("/home/fproenca/Tese/Data/CUHK01/labels.pickle","rb")
  labels = pickle.load(pickle_in)
  x = x[:3484]
  labels = labels[:3484]
  return x,labels

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
      for idxB in range(1):
        augmented_image = data_augmentation(image)
        aug_img.append(augmented_image[0])
        labels_aug.append(label)
  print(len(labels_aug))
  return np.array(aug_img),np.array(labels_aug)

x, labels = load_data()

x_val = x[3080:]
labels_val = labels[3080:]
x_train = x[:3080]
labels_train = labels[:3080]

aug_img,aug_label = aug_data(x_train,labels_train)
x_train = np.concatenate((x_train, aug_img))
labels_train = np.concatenate((labels_train, aug_label))

app = '/home/fproenca/Tese/junk/'

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


print(len(labels_train))
print(len(labels_val))
print(len(x_val))
print(len(x_train))

x_val = x_val_
x_train = x_train_

def make_pairs(images,labels,flag):
  # initialize two empty lists to hold the (image, image) pairs and
	# labels to indicate if a pair is positive or negative
  pairImages = []
  pairLabels = []
	# calculate the total number of classes present in the dataset
	# and then build a list of indexes for each class label that
	# provides the indexes for all examples with a given label
  numClasses = len(np.unique(labels))
  idx = [np.where(labels == i)[0] for i in range(0, numClasses+1)]
  #print(numClasses)
  #print(idx)
  cnt = 0
  aux = np.zeros(4000)
  stri = '.npy'
  app = '/home/fproenca/Tese/junk/'
  if flag == 1:
    stri = '_.npy'
  for idxA in range(len(images)-1):
      # grab the current image and label belonging to the current
      # iteration
      currentImage = images[idxA]
      label = labels[idxA]
      for idxB in range(idxA+1,len(images)):
        posImage = images[idxB]
        label_ = labels[idxB]
        aux[label_] = aux[label_]+1
        if (label == label_):
          #print("Pos1: " + str(idxA) + "Pos2: " + str(idxB))
          pairImage = [currentImage, posImage]
          name = app + str(cnt) + stri
          pairImage = np.save(name,pairImage)
          pairImages.append(name)
          pairLabels.append([1])
          cnt +=1
          negIdx = np.where(labels != label)[0]
          negImage = images[np.random.choice(negIdx)]
		      # prepare a negative pair of images and update our lists
          pairImage = [currentImage, negImage]
          name = app + str(cnt) + stri
          pairImage = np.save(name,pairImage)
          cnt +=1
          pairImages.append(name)
          pairLabels.append([0])          
	# return a 2-tuple of our image pairs and labels 
  return pairImages, pairLabels


# make train pairs
pairs_train, labels_train = make_pairs(x_train, labels_train,0)

# make validation pairs
pairs_val, labels_val = make_pairs(x_val, labels_val,1)


def fetch(batch_x):
  pair_file = []
  pairImages = []
  #print(batch_x)
  for file_name in batch_x:
    pair_file = (np.load(file_name))
    pair = [np.load(pair_file[0]) , np.load(pair_file[1])]
    pairImages.append(pair)
  pairImages = np.array(pairImages)
  data = ([pairImages[:,0],pairImages[:,1]])
  
  return data


def fetchy(batch_y):
  #print(batch_y)
  pairImages = np.array(batch_y)
  return pairImages


class My_Custom_Generator(tf.keras.utils.Sequence) :
  
  def __init__(self, image_filenames, labels, batch_size) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return fetch(batch_x) , fetchy(batch_y).astype("float32")

batch_size = 32

pairs_train, labels_train = shuffle(pairs_train, labels_train)
pairs_val, labels_val = shuffle(pairs_val, labels_val)
print(" Pairs: " + str(len(pairs_train)))
my_training_batch_generator = My_Custom_Generator(pairs_train, labels_train, batch_size)
my_validation_batch_generator = My_Custom_Generator(pairs_val, labels_val, batch_size)

siamese = model_obt()



def loss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

  Arguments:
      margin: Integer, defines the baseline for distance for which pairs
              should be classified as dissimilar. - (default is 1).

  Returns:
      'constrastive_loss' function with data ('margin') attached.
  """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

      Arguments:
          y_true: List of labels, each label is of type float32.
          y_pred: List of predictions of same length as of y_true,
                  each label is of type float32.

      Returns:
          A tensor containing constrastive loss as floating point value.
      """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss


lr_schedule = tensorflow.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=1000000,
    decay_rate=0.9)

opt = tensorflow.keras.optimizers.SGD(learning_rate=lr_schedule)

siamese.compile(loss=loss(margin=margin), optimizer=opt, metrics=["accuracy"])
siamese.summary()

callbacks = ModelCheckpoint('/home/fproenca/Tese/Results/CUHK01/final_test_match.h5', monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True, mode='auto')

siamese.load_weights('/home/fproenca/Tese/Results/CUHK01/final_test_match.h5')


history = siamese.fit(
    my_training_batch_generator,
    epochs = 200,
    callbacks = callbacks,
    verbose = 1,
    validation_data = my_validation_batch_generator,
)



























