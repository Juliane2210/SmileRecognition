#
#  Juliane Bruck  - 8297746
#  Assignment #2
#
# Based on https://github.com/cocoxu/SemEval-PIT2015
# Data located at https://drive.google.com/file/d/1ka0c6LqDLwovePRRzeuJFLx7AcXcz8Ji/view?usp=drive_link
# Please download and point the path at the data prior to running
#
import os
import cv2
import shutil
import pandas as pd
from sklearn.metrics import precision_score, recall_score, mean_squared_error
from sklearn.metrics import roc_curve, precision_recall_curve, auc, precision_score, recall_score


from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from skimage.feature import hog
from skimage import exposure
from skimage.feature import local_binary_pattern
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score


# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define constants
IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100
BATCH_SIZE = 32


def grayscale_conversion(img):
    # Convert the image to grayscale
    # Add a color channel axis to make it compatible with the original shape
    grayscale_img = resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    grayscale_img = np.expand_dims(grayscale_img, axis=-1)
    return grayscale_img

# Function to preprocess the input image
# def preprocess_image(image_path):
#     img = image.load_img(image_path, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
#     img_array = image.img_to_array(img)
#     # Expand dimensions to create batch dimension
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0  # Normalize pixel values
#     return img_array


def createModel():
    # Set up data generators for training, validation, and testing
    train_data_generator = ImageDataGenerator(rescale=1./255,
                                            #   preprocessing_function=grayscale_conversion,  # Apply grayscale conversion
                                              validation_split=0.2,
                                              )
    train_generator = train_data_generator.flow_from_directory(
        ".\\data\\testimages"
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True,
        color_mode='grayscale',
        subset="training",
        seed=42
    )

    validation_generator = train_data_generator.flow_from_directory(
        ".\\data\\testimages",
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True,
        color_mode='grayscale',
        subset="validation",
        seed=42
    )
    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                               input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer="adam", loss='binary_crossentropy',
                  metrics=['accuracy'])
    # Create a ModelCheckpoint callback to save the best model
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        'champion_model_drop_0_05.keras', 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max', 
        verbose=1
    )

    # Train the model with the checkpoint callback
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[checkpoint_callback]
    )

    return model


if __name__ == "__main__":
    print('Creating model')
    model = createModel()
    print('Saving model')
    model.save("smile_detection_model.keras")
    print("All done!")
