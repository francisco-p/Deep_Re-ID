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
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def model_obt():
    model = load_model('/home/fproenca/Tese/Results/CUHK01/MobileNet_Class_CUHK01_224.h5')
    #model.summary()
    embedding_network = Sequential(name="Embedding")
    for layer in model.layers[:-2]: # go through until last layer
        embedding_network.add(layer)
    for layer in embedding_network.layers:
      if layer.name == "Embedding":
        for i in layer.layers:
          #print(i.name)
          i.trainable = False
    #embedding_network.add((Dense(1024, activation='relu')))
    #embedding_network.add(Dropout(0.3))
    #embedding_network.add((Dense(512, activation='relu',name="512")))
    #embedding_network.add((Dense(256, activation='relu',name ='123')))

    trainable = True
    for layer in embedding_network.layers:
      #print(layer.name == "mobilenet_1.00_224")
      if layer.name == "mobilenet_1.00_224":
        for i in layer.layers:
          #print(i.name == "block_16_expand")
          if i.name == "block_16_expand":
            trainable = True
          i.trainable = trainable
    for layer in embedding_network.layers:
      if layer.name == 'mobilenet_1.00_224':
        for i in layer.layers:
          print(str(i.name) + " " + str(i.trainable))  
    #for layer in embedding_network.layers:
      #print(layer.trainable)
      #layer.trainable = False
    

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

    return siamese
