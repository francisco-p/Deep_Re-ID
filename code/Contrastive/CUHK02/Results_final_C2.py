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


#####################################
# Provided two tensors t1 and t2
# Euclidean distance = sqrt(sum(square(t1-t2)))
def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tensorflow.math.reduce_sum(tensorflow.math.square(x - y), axis=1, keepdims=True)
    return tensorflow.math.sqrt(tensorflow.math.maximum(sum_square, tensorflow.keras.backend.epsilon()))



model = load_model('/home/fproenca/Tese/Results/CUHK02/MobileNet_Class_224.h5')
model.summary()
embedding_network = Sequential(name="Embedding")
for layer in model.layers[:-2]: # go through until last layer
    embedding_network.add(layer)
embedding_network.summary()
trainable = True
for layer in embedding_network.layers:
  if layer.name == "mobilenet_1.00_224":
    for i in layer.layers:
      if i.name == "conv1":
        trainable = True
      i.trainable = trainable
for layer in embedding_network.layers:
  if layer.name == 'mobilenet_1.00_224':
    for i in layer.layers:
      print(str(i.name) + " " + str(i.trainable))
embedding_network.summary()

input_1 = tensorflow.keras.layers.Input((224, 224, 3))
input_2 = tensorflow.keras.layers.Input((224, 224, 3))

# As mentioned above, Siamese Network share weights between
# tower networks (sister networks). To allow this, we will use
# same embedding network for both tower networks.
tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)

merge_layer = tensorflow.keras.layers.Lambda(euclidean_distance)([tower_1, tower_2])
normal_layer = tensorflow.keras.layers.BatchNormalization()(merge_layer)
output_layer = tensorflow.keras.layers.Dense(1, activation="sigmoid")(normal_layer)
siamese = tensorflow.keras.Model(inputs=[input_1, input_2], outputs=output_layer)
###################################



siamese.load_weights('/home/fproenca/Tese/Results/CUHK02/final_test_match.h5')
print('/home/fproenca/Tese/Results/CUHK02/final_test_match.h5')
siamese.summary()


# Load all data even the test data
pickle_in = open("/home/fproenca/Tese/Data/CUHK02/x_total.pickle","rb")
x = pickle.load(pickle_in)
pickle_in = open("/home/fproenca/Tese/Data/CUHK02/labels_total.pickle","rb")
labels = pickle.load(pickle_in)


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

def euclidean_distance_(querys, gallery, topk):
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
            dist = siamese.predict([np.expand_dims(q_feature, axis=0), np.expand_dims(feature, axis=0)])
            distmat.append([dist, label])
        # Sort the results for each query
        distmat.sort(reverse= True)
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

for j in range(1, 1817):
  aux_x = []
  aux_labels = []
  for i in range(len(labels)):
    if labels[i] == j:
      aux_x.append(x[i])
      aux_labels.append(labels[i])
  x_classes.append(aux_x)
  labels_classes.append(aux_labels)

for i in range(1816):
  gallery_list.append([labels_classes[i][2], x_classes[i][2]])
  gallery_list.append([labels_classes[i][3], x_classes[i][3]])



randomlist = random.sample(range(1716, 1816), 100)

for i in randomlist:
  random.seed(i+1)
  rand = random.randint(0,1)
  query_list.append([labels_classes[i][rand], x_classes[i][rand]])


cmc_re, sum_ranks, map = euclidean_distance_(query_list, gallery_list, len(gallery_list))

print("2048 -> Rank1: " + str(cmc_re[0]) + " Rank_5: " + str(cmc_re[4]) + " Rank_10: " + str(cmc_re[9]) + " Rank_20: " + str(cmc_re[19]))
print(cmc_re[:50])
print(map)
map_2048 = calculate_map(sum_ranks, len(query_list))
print(map_2048)






