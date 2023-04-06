"""
Patrick Guo
101109793
Eric Pham
101104095
George Li
101107279
COMP 4107
Matthew Holden
April 12, 2023
Project Code Portion
"""
import os, os.path
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sklearn
from sklearn.model_selection import train_test_split
import preprocessing


IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64


class ImageDataGenerator(tf.keras.utils.Sequence):
  """
  Keras data generator for custom image dataset
  """

  def __init__(self, image_directory, batch_size=64, preprocess=False):
    """
    Initialize data generator for hill valley data.

    :input dataset_filepath: Path to a .csv file containing the dataset
    :input use_rows: Subset of the rows in the file to be used
    :input batch_size: Batch size for the network
    """
    self.batch_size = batch_size
    self.preprocess = preprocess

    self.split_ids = []

    self.non_ai_directory = os.path.join(image_directory, 'non_ai')
    self.ai_directory = os.path.join(image_directory, 'ai')

    # Add non-AI image ID and label 0
    self.non_ai_size = 0
    for name in os.listdir(self.non_ai_directory):
      if os.path.isfile(os.path.join(self.non_ai_directory, name)):
        self.split_ids.append((name, 0))
        self.non_ai_size += 1

    # Add AI image ID and label 1
    self.ai_size = 0
    for name in os.listdir(self.ai_directory):
      if os.path.isfile(os.path.join(self.ai_directory, name)):
        self.split_ids.append((name, 1))
        self.ai_size += 1

    self.on_epoch_end() # Shuffle labels

  def __len__(self):
    """
    Get number of batches used for one epoch.

    :return: Total number of batches used for one epoch
    """
    batches_per_epoch = int((self.non_ai_size + self.ai_size) / self.batch_size)
    return batches_per_epoch

  def __getitem__(self, index):
    """
    Get a batch of data and its labels at the given index.

    :input index: Index of the batch to be retrieved
    :return: Tuple of batch of data and asociated labels
    """
    batch_ids = self.split_ids[index * self.batch_size:(index + 1) * self.batch_size]
    
    # Generate data
    x = None
    y = None
    for file_name, label in batch_ids:
      
      if label == 0: # non-AI image
        image_filename = os.path.join(self.non_ai_directory, file_name)
      else: # AI image
        image_filename = os.path.join(self.ai_directory, file_name)
    
      image = Image.open(image_filename).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
      if self.preprocess is False and image.mode != 'RGB':
        image = image.convert('RGB')
      elif self.preprocess is not False:
        image = image.convert("L")

      image_array = np.asarray(image)

      # Preprocessing
      if self.preprocess == 'edge':
        edge_mag, edge_ori = preprocessing.edge_mag_ori(image_array, 1)
        image_array = edge_mag
      elif self.preprocess == 'orient':
        edge_mag, edge_ori = preprocessing.edge_mag_ori(image_array, 1)
        image_array = edge_ori
      elif self.preprocess == 'corner':
        image_array = preprocessing.corner_detect(image_array)

      image_array = np.expand_dims(np.asarray(image), axis=0)

      if x is None:
        x = image_array
        y = label
      else:
        x = np.vstack((x, image_array))
        y = np.vstack((y, label))

    # x is one batch of data
    # y is the same batch of data
    return x, y
  
  def on_epoch_end(self):
    np.random.shuffle(self.split_ids)


def ai_image_detector_model(training_data_path, val_data_path, preprocess):
  """
  Creates a keras CNN model to predict whether an image is AI generated or not.

  :input training_data_filepath: Path to a directory containing the training dataset
  :input validation_data_filepath: Path to a directory containing the validation dataset
  :return: Keras model, training performance, validation performance
  """
  numChannels = 3 if preprocess == False else 1

  model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, numChannels)),
    keras.layers.Conv2D(filters=32, kernel_size=3, activation='sigmoid'),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(filters=64, kernel_size=3, activation='sigmoid'),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(filters=64, kernel_size=3, activation='sigmoid'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="sigmoid"),
    keras.layers.Dense(1, activation="sigmoid")
  ])

  print(model.summary())

  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
  model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

  trainval_generator = ImageDataGenerator(training_data_path, preprocess=preprocess)
  validate_generator = ImageDataGenerator(val_data_path, preprocess=preprocess)
  training = model.fit(x=trainval_generator, validation_data=validate_generator, epochs=10, verbose=1)

  training_performance = training.history['loss'][-1]
  validation_performance = training.history['val_loss'][-1]

  return model, training_performance, validation_performance


if __name__ == '__main__':
  # Testing code
  dataset_train_fullpath = r"H:\Downloads\School\Year5\comp4107\project\test\test_dataset\train"
  dataset_val_fullpath = r"H:\Downloads\School\Year5\comp4107\project\test\test_dataset\validation"
  dataset_test_fullpath = r"H:\Downloads\School\Year5\comp4107\project\test\test_dataset\test"
  #preprocess = False
  preprocess = 'edge' # Can be: False, 'edge', 'orient', or 'corner'

  # Data gen
  '''datagen = ImageDataGenerator(dataset_test_fullpath, batch_size=100)
  print(datagen.__len__())
  batch_1 = datagen.__getitem__(0)
  print("batch 1:", batch_1[0].shape, batch_1[1].shape)
  batch_2 = datagen.__getitem__(1)
  print("batch 2:", batch_2[0].shape, batch_2[1].shape)
  batch_3 = datagen.__getitem__(2)
  print("batch 3:", batch_3[0].shape, batch_3[1].shape)
  batch_4 = datagen.__getitem__(3)
  print("batch 4:", batch_4[0].shape, batch_4[1].shape)
  batch_5 = datagen.__getitem__(4)
  print("batch 5:", batch_5[0].shape, batch_5[1].shape)
  batch_6 = datagen.__getitem__(5)
  print("batch 6:", batch_6[0].shape, batch_6[1].shape)
  batch_7 = datagen.__getitem__(6)
  print("batch 7:", batch_7[0].shape, batch_7[1].shape)
  batch_8 = datagen.__getitem__(7)
  print("batch 8:", batch_8[0].shape, batch_8[1].shape)
  batch_9 = datagen.__getitem__(8)
  print("batch 9:", batch_9[0].shape, batch_9[1].shape)
  batch_10 = datagen.__getitem__(9)
  print("batch 10:", batch_10[0].shape, batch_10[1].shape)
  batch_11 = datagen.__getitem__(10)
  print("batch 11:", batch_11[0].shape, batch_11[1].shape)
  batch_12 = datagen.__getitem__(11)
  print("batch 12:", batch_12[0].shape, batch_12[1].shape)'''

  # Model
  model, training_performance, validation_performance = ai_image_detector_model(dataset_train_fullpath, dataset_val_fullpath, preprocess)
  print("CNN Training loss:", training_performance)
  print("CNN Validation loss:", validation_performance)

  test_generator = ImageDataGenerator(dataset_test_fullpath, preprocess=preprocess)
  test_performance = model.evaluate(x=test_generator)
  print("CNN Test loss:", test_performance)
