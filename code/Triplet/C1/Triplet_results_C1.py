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

######################################################################################
# Load all data even the test data
pickle_in = open(path_data1_total,"rb")
x = pickle.load(pickle_in)

pickle_in = open(path_data2_total,"rb")
labels = pickle.load(pickle_in)
######################################################################################
######################################################################################

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

######################################################################################
######################################################################################

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


siamese_network.load_weights(emb_name)

embedding_new = Sequential(name="Embedding_new") 

for layer in embedding.layers[:-1]: # go through until last layer
    embedding_new.add(layer)
embedding_new.summary()

def get_mobile_net(x):
	
	feature_vectors = []
	i=0
	for xi in x:
	  feature_vectors.append(embedding_new(np.expand_dims(((xi)), axis=0)))
	  #feature_vectors.append(embedding(np.expand_dims(((xi)), axis=0)))
	  i+=1
	
	for j in range(len(feature_vectors)):
		feature_vectors[j] = feature_vectors[j][0]
	
	feature_vectors = np.array(feature_vectors)

	return feature_vectors

#Obtain feature vectors
feat_vect = get_mobile_net(x)
######################################################################################

pickle_out = open( '/home/fproenca/Tese/Data/CUHK01/feat_new.pickle',"wb")
pickle.dump(feat_vect, pickle_out)
pickle_out.close()

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
            #cosine_similarity = metrics.CosineSimilarity()
            #cos = cosine_similarity(q_feature, feature)
            #dist = cos.numpy()
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
