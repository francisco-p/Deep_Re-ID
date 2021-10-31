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
path_data1 = "/home/fproenca/Tese/Data/CUHK01/x.pickle"
path_data2 = "/home/fproenca/Tese/Data/CUHK01/labels.pickle"
app = '/home/fproenca/Tese/junk/'
model_name = '/home/fproenca/Tese/Results/CUHK01/MobileNet_Class_CUHK01_224.h5'
call_name = '/home/fproenca/Tese/Results/CUHK01/triplet.h5'
path_data1_total = "/home/fproenca/Tese/Data/CUHK01/x_total.pickle"
path_data2_total = "/home/fproenca/Tese/Data/CUHK01/labels_total.pickle"
emb_name = '/home/fproenca/Tese/Results/CUHK01/emb_tri.h5'
triplet_train_name = '/home/fproenca/Tese/Data/CUHK01/triplet_train.pickle'
triplet_val_name = '/home/fproenca/Tese/Data/CUHK01/triplet_val.pickle'

##################################################################################

def load_data():
  pickle_in = open(path_data1,"rb")
  x = pickle.load(pickle_in)
  pickle_in = open(path_data2,"rb")
  labels = pickle.load(pickle_in)
  return x,labels


x, labels = load_data()

x_val = x[3080:]
labels_val = labels[3080:]
x_train = x[:3080]
labels_train = labels[:3080]


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
#embedding.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
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

    def __init__(self, siamese_network, margin=15):
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
opt= optimizers.Adam(0.000001)
#opt = tensorflow.keras.optimizers.SGD(learning_rate= lr_schedule)
siamese_model.compile(optimizer=opt)
callbacks = ModelCheckpoint(emb_name, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')


siamese_model.fit(
    my_training_batch_generator,
    epochs = 50,
    steps_per_epoch = 500,
    verbose = 1,
    callbacks = callbacks,
    validation_data = my_validation_batch_generator,    
)
######################################################################################
# Load all data even the test data
pickle_in = open(path_data1_total,"rb")
x = pickle.load(pickle_in)

pickle_in = open(path_data2_total,"rb")
labels = pickle.load(pickle_in)

def get_mobile_net(x):
	
	feature_vectors = []
	i=0
	for xi in x:
	  feature_vectors.append(embedding(np.expand_dims(((xi)), axis=0)))
	  i+=1
	
	for j in range(len(feature_vectors)):
		feature_vectors[j] = feature_vectors[j][0]
	
	feature_vectors = np.array(feature_vectors)

	return feature_vectors

#Obtain feature vectors
feat_vect = get_mobile_net(x)
######################################################################################

def AP_calculation(rank_vector,number_matches):
  prev = 1/number_matches
  sum = 0
  cnt = 0
  for idx, a in enumerate(rank_vector):
    if a == 0:
      continue
    else:
      cnt += 1
      sum = sum + cnt/(idx+1)
  AP = prev * sum
  return AP
  
def map_calculate(ap_vector, query_number):
  prev = 1 / query_number
  af = np.sum(ap_vector)

  mAP = prev * af
  return mAP

  #Calculate mAP
def calculate_map(sum_ranks, n):
  sum = 0
  for i in range(len(sum_ranks)):
    sum += sum_ranks[i]/(i+1)
  map = (1/n)*sum
  print(map)

  return map

def euclidean_distance(querys, gallery, topk):
    aux = 0
    valid_queries = 0
    all_rank = []
    all_rank_map = []
    sum_rank = np.zeros(topk)
    for query in querys:
        aux += 1
        print(aux)
        q_id = query[0]
        q_feature = query[1]
        # Calculate the distances for each query
        distmat = []
        for label, feature in gallery:
            dist = np.linalg.norm(q_feature - feature)
            distmat.append([dist, label])
        # Sort the results for each query
        distmat.sort()
        # Find matches
        matches = np.zeros(len(distmat))
        # Zero if no match 1 if match
        for i in range(0, len(distmat)):
            if distmat[i][1] == q_id:
                # Match found
                matches[i] = 1
        rank = np.zeros(topk)
        rank_map = np.zeros(topk)
        for i in range(0, topk):
            if matches[i] == 1:
                rank_map[i] = 1
        for i in range(0, topk):
            if matches[i] == 1:
                rank[i] = 1
                # If 1 is found then break as you dont need to look further path k
                break
        valid_queries +=1
        all_rank.append(rank)
        all_rank_map.append(rank_map)
    for i in all_rank_map:
        print ("-->" + str(i[0]) + " " + str(i[1]) + " " + str(i[2]) + " " + str(i[3]) + " " + str(i[4]) + " " + str(i[5]) + " " + str(i[6]) + " " + str(i[7]) + " " + str(i[8]) + " " + str(i[9]) + " " + str(i[10]) + " " + str(i[11]) + " " + str(i[12]) + " " + str(i[13]) + " " + str(i[14]) + " " + str(i[15]) + " "  )
    print("............................................")
    ## CMC curve - Rank results ##
    sum_all_ranks = np.zeros(len(all_rank[0]))
    for i in range(0,len(all_rank)):
        my_array = all_rank[i]
        for g in range(0, len(my_array)):
            sum_all_ranks[g] = sum_all_ranks[g] + my_array[g]
    sum_all_ranks = np.array(sum_all_ranks)
    print("NPSAR", sum_all_ranks)
    cmc_restuls = np.cumsum(sum_all_ranks) / 100
    print(cmc_restuls)
    ##  mAP calculation ##
    AP = np.zeros(len(all_rank_map))
    for i in range(0,len(all_rank_map)):
        my_array = all_rank_map[i]
        # Change if not single gallery shot and not 100 queries#
        AP[i] = AP_calculation(my_array,2)
    map = map_calculate(AP, 100)
    

    return cmc_restuls, sum_all_ranks, map

query_list = []
gallery_list = []

x_classes = []
labels_classes = []

for j in range(1, 972):
  aux_x = []
  aux_labels = []
  for i in range(len(labels)):
    if labels[i] == j:
      aux_x.append(feat_vect[i])
      aux_labels.append(labels[i])
  x_classes.append(aux_x)
  labels_classes.append(aux_labels)

for i in range(971):
  gallery_list.append([labels_classes[i][2], x_classes[i][2]])
  gallery_list.append([labels_classes[i][3], x_classes[i][3]])


randomlist = random.sample(range(871, 971), 100)

for i in randomlist:
  random.seed(i+0)
  rand = random.randint(0,1)
  query_list.append([labels_classes[i][rand], x_classes[i][rand]])

cmc_re, sum_ranks, map = euclidean_distance(query_list, gallery_list, len(gallery_list))

print("2048 -> Rank1: " + str(cmc_re[0]) + " Rank_5: " + str(cmc_re[4]) + " Rank_10: " + str(cmc_re[9]) + " Rank_20: " + str(cmc_re[19]))
print(cmc_re[:20])
print(map)
map_2048 = calculate_map(sum_ranks, len(query_list))
print(map_2048)
