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

epochs = 10
batch_size = 16
margin = 1  # Margin for constrastive loss.


def load_data():
  pickle_in = open("/home/fproenca/Tese/Data/HDA/x.pickle","rb")
  x = pickle.load(pickle_in)

  pickle_in = open("/home/fproenca/Tese/Data/HDA/labels.pickle","rb")
  labels = pickle.load(pickle_in)
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


x_val = x[9299:]
labels_val = labels[9299:]
x_train = x[:9299]
labels_train = labels[:9299]


#aug_img,aug_label = aug_data(x_train,labels_train)
#x_train = np.concatenate((x_train, aug_img))
#labels_train = np.concatenate((labels_train, aug_label))


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
  counter_max = 0
  aux_max = 0
  for idxA in range(len(images)-1):
      # grab the current image and label belonging to the current
      # iteration
      currentImage = images[idxA]
      label = labels[idxA]
      if label == aux_max:
         counter_max += 1
      else:
         counter_max = 0
      aux_max = label
      if counter_max > 12:
       continue
      #print("Label:" + str(label) + " || Aux_MAX" + str(aux_max) +" || Counter_Max" + str(counter_max))

      #print(label)
      for idxB in range(idxA+1,len(images)):
        posImage = images[idxB]
        label_ = labels[idxB]
        
        #print(label_)
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

print("Hereeeeeee")
print(len(pairs_train))
print(len(pairs_val))


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

#labels_train = labels_train[:150000]
#pairs_train = pairs_train[:150000]
#pairs_val = pairs_val[:300000]
#labels_val = labels_val[:300000]

print(len(pairs_train))
print(len(pairs_val))
my_training_batch_generator = My_Custom_Generator(pairs_train, labels_train, batch_size)
my_validation_batch_generator = My_Custom_Generator(pairs_val, labels_val, batch_size)


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


model = load_model('/home/fproenca/Tese/Results/HDA/MobileNet_Class_224.h5')
model.summary()
embedding_network = Sequential(name="Embedding")
for layer in model.layers[:-2]: # go through until last layer
    embedding_network.add(layer)
embedding_network.summary()
trainable = False
for layer in embedding_network.layers:
  #print(layer.name == "mobilenet_1.00_224")
  if layer.name == "mobilenet_1.00_224":
    for i in layer.layers:
      #print(i.name == "block_16_expand")
      if i.name == "conv1":
        trainable = True
      i.trainable = trainable
for layer in embedding_network.layers:
  if layer.name == 'mobilenet_1.00_224':
    for i in layer.layers:
      print(str(i.name) + " " + str(i.trainable))
#embedding_network.add(Dense(512, activation='relu',name = "512"))
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

#for layer in siamese.layers:
#  if layer.name == "Embedding":
#    for i in layer.layers:
#      print(i.name)
#      i.trainable = False

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
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)

opt = tensorflow.keras.optimizers.SGD(learning_rate=lr_schedule)

siamese.compile(loss=loss(margin=margin), optimizer=opt, metrics=["accuracy"])
siamese.summary()

callbacks = ModelCheckpoint('/home/fproenca/Tese/Results/HDA/final_test_match.h5', monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True, mode='auto')

history = siamese.fit(
    my_training_batch_generator,
    epochs = 250,
    callbacks = callbacks,
    verbose = 1,
    validation_data = my_validation_batch_generator,    
)