

import os
import cv2
import shutil
import sys
import pandas as pd
from sklearn.metrics import precision_score, recall_score, mean_squared_error
from sklearn.metrics import roc_curve, precision_recall_curve, auc, precision_score, recall_score
from tensorflow.keras.models import load_model

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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

# Set seeds for NumPy and TensorFlow random number generators
np.random.seed(256)
tf.random.set_seed(256)

#  true image dimensions are (162, 193) `rows` is not in [96, 128, 160, 192, 224]. Weights for input shape (224, 224)
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

# Make predictions or perform other operations with the loaded model

# Define a function to convert images to grayscale


def grayscale_conversion(img):
    # Convert the image to grayscale
    grayscale_img = rgb2gray(img)
    # Add a color channel axis to make it compatible with the original shape
    grayscale_img = np.expand_dims(grayscale_img, axis=-1)
    return grayscale_img


def getImageDataGenerator():
    return ImageDataGenerator(
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        # rescale=None,
        rescale=1./255,
        data_format=None,
        validation_split=0.0,
        interpolation_order=1,
        dtype=None,
        preprocessing_function=grayscale_conversion  # Apply grayscale conversion
    )


def getPredictions(model, threshold, file_path):

    prediction_datagen = getImageDataGenerator()

    to_predict_generator = prediction_datagen.flow_from_directory(file_path,

                                                                  target_size=(
                                                                      IMAGE_WIDTH, IMAGE_HEIGHT),
                                                                  batch_size=32,
                                                                  class_mode='categorical'
                                                                  )
    y_true = to_predict_generator.classes

    # filepaths = to_predict_generator.filepaths

    # Get the predicted probabilities for each image
    y_pred_probs = model.predict(to_predict_generator)

    # Convert probabilities to binary predictions (0 or 1) based on a threshold
    # y_pred = np.where(y_pred_probs > 0.5, 1, 0)

    class_indices = to_predict_generator.class_indices
    # Extract elements where the second item is greater than the first
    y_pred = [1 if item[class_indices['1']] >
              threshold else 0 for item in y_pred_probs]

    return y_true, y_pred, to_predict_generator


#
# Visualizations
# Helper functions that will be called to perform visualizations of model evaluation scores
#

def visualize_predictions_confusionMatrix(title, true_labels, predicted_labels):
    # Plot confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[
                '0', '1'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title + ": Confusion Matrix")

    plt.show()


def visualize_predictions_ROC(title, true_labels, predicted_labels):
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title + ': Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def visualize_predictions_PrecisionRecall(title, true_labels, predicted_labels):
    # Calculate Precision-Recall curve
    precision, recall, _ = precision_recall_curve(
        true_labels, predicted_labels)
    pr_auc = auc(recall, precision)

    # Plot Precision-Recall curve
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2,
             label='Precision-Recall curve (area = %0.2f)' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title + ': Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()


def analyzePredictions(y_true, y_pred, to_predict_generator):

    visualize_predictions_confusionMatrix("test data", y_true, y_pred)
    visualize_predictions_ROC("test data", y_true, y_pred)
    visualize_predictions_PrecisionRecall("test data", y_true, y_pred)
# Assuming y_true and y_pred are your true labels and predicted labels respectively
    f1 = f1_score(y_true, y_pred)

    print("F1 Score:", f1)
    #
    # The code below outputs the converted images so we can verify what is being compared.
    #

    # output_folder_happy = ".\\data\\test_results\\1\\"
    # output_folder_neutral = ".\\data\\test_results\\0\\"
    # output_file_path = ".\\data\\test_results\\"

    # # Ensure that the output folder exists
    # os.makedirs(output_folder_happy, exist_ok=True)
    # os.makedirs(output_folder_neutral, exist_ok=True)

    # # Iterate over the generator to process and save each image
    # for i, (images, _) in enumerate(to_predict_generator):
    #     for j, image in enumerate(images):
    #         # Generate a file name for the image
    #         filename = f"image_{i * to_predict_generator.batch_size + j}.jpg"

    #         subfolder = y_pred[j]
    #         # Save the image to the output directory
    #         image_path = os.path.join(output_file_path, filename)
    #         # Convert back to uint8 before saving
    #         Image.fromarray((image * 255).astype(np.uint8)).save(image_path)
    #     # Stop iteration after processing all batches
    #     if i + 1 == len(to_predict_generator):
    #         break

    # print("All images processed and saved to:", output_file_path)


if __name__ == "__main__":

    # Check if the file path argument is provided
    if len(sys.argv) != 2:
        print("Usage: python predict.py <file_path>")
        file_path = ".\\data\\testing_images"
        # sys.exit(1)
    else:
        # Extract file path from command-line arguments
        file_path = sys.argv[1]

    # Load the model
    model = load_model(".\\smile_detection_model.keras")

    y_true, y_pred, to_predict_generator = getPredictions(
        model, 0.6, file_path)
    analyzePredictions(y_true, y_pred, to_predict_generator)
