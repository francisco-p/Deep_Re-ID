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
path_data1 = "/home/fproenca/Tese/Data/Market/x.pickle"
path_data2 = "/home/fproenca/Tese/Data/Market/labels.pickle"
app = '/home/fproenca/Tese/M/junk/'
model_name = '/home/fproenca/Tese/Results/Market/MobileNet_Class_224.h5'
call_name = '/home/fproenca/Tese/Results/Market/triplet.h5'
path_data1_total = "/home/fproenca/Tese/Data/Market/x_total.pickle"
path_data2_total = "/home/fproenca/Tese/Data/Market/labels_total.pickle"
emb_name = '/home/fproenca/Tese/Results/Market/emb_tri.h5'
triplet_train_name = '/home/fproenca/Tese/Data/Market/triplet_train.pickle'
triplet_val_name = '/home/fproenca/Tese/Data/Market/triplet_val.pickle'
train_loss = "/home/fproenca/Tese/Data/Market/train_loss.pickle"
val_loss = "/home/fproenca/Tese/Data/Market/val_loss.pickle"
aug_val = 1
split_val = 11763
margin = 0.5

##################################################################################

def load_data():
  pickle_in = open(path_data1,"rb")
  x = pickle.load(pickle_in)
  pickle_in = open(path_data2,"rb")
  labels = pickle.load(pickle_in)
  return x,labels


x, labels = load_data()

x_val = x[split_val:]
labels_val = labels[split_val:]
x_train = x[:split_val]
labels_train = labels[:split_val]


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
  if l + 0.4 > 0.0:
   tri_final.append(tri)
 return(np.array(tri_final))
  
triplet_train = prepare_triplets(triplet_train,train_loss)
triplet_val = prepare_triplets(triplet_val,val_loss)

print(len(triplet_train))
print(len(triplet_val))


def fetch(batch_x):
  
  tripletImages = []
  for file_name in batch_x:
    triplet_file = (np.load(file_name))
    w = random.randint(0,1)
    triplet = [np.load(triplet_file[1-w]) , np.load(triplet_file[w]),np.load(triplet_file[2])]
    tripletImages.append(triplet)
  tripletImages = np.array(tripletImages)
  #data = ([tripletImages[:,0],tripletImages[:,1],tripletImages[:,2]])
  return [tripletImages[:,0],tripletImages[:,1],tripletImages[:,2]]
class My_Custom_Generator(tf.keras.utils.Sequence) :
  
  def __init__(self, image_filenames, batch_size) :
    self.image_filenames = image_filenames
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return ([fetch(batch_x)])

batch_size = 16
tripletImages_train = shuffle(triplet_train)
tripletImages_val = shuffle(triplet_val)
my_training_batch_generator = My_Custom_Generator(tripletImages_train, batch_size)
my_validation_batch_generator = My_Custom_Generator(tripletImages_val, batch_size)

#############################################################

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
#embedding.add(Dense(128, activation='relu',name = 'Dense1'))
embedding.summary()

############################################################
class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

anchor_input = tensorflow.keras.layers.Input(name="anchor", shape= (input,input ,3))
positive_input = tensorflow.keras.layers.Input(name="positive", shape= (input,input ,3))
negative_input = tensorflow.keras.layers.Input(name="negative", shape= (input,input ,3))

distances = DistanceLayer()(
    embedding(anchor_input),
    embedding(positive_input),
    embedding(negative_input),
)

siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)
siamese_network.summary()

class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
           """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]

siamese_model = SiameseModel(siamese_network)


#lr_schedule = tensorflow.keras.optimizers.schedules.ExponentialDecay(
 #   initial_learning_rate=0.0001,
  #  decay_steps=10000,
   # decay_rate=0.9)
opt= optimizers.Adam(0.0000005)
#opt = tensorflow.keras.optimizers.SGD(learning_rate= lr_schedule)
siamese_model.compile(optimizer=opt)
callbacks = ModelCheckpoint(emb_name, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')





siamese_model.fit(
    my_training_batch_generator,
    epochs = 50,
    steps_per_epoch = 6000,
    validation_steps = 2000,
    verbose = 1,
    callbacks = callbacks,
    validation_data = my_validation_batch_generator,    
)

