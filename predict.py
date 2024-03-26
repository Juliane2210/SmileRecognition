

import os
import sys
from sklearn.metrics import roc_curve, precision_recall_curve, auc, precision_score, recall_score
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report
from transformers import AutoTokenizer
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

# Set seeds for NumPy and TensorFlow random number generators
# np.random.seed(256)
# tf.random.set_seed(256)

#  true image dimensions are (162, 193) `rows` is not in [96, 128, 160, 192, 224]. Weights for input shape (224, 224)
IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100

BATCH_SIZE = 32

# Make predictions or perform other operations with the loaded model

# Define a function to convert images to grayscale


def grayscale_conversion(img):
    # Convert the image to grayscale
    grayscale_img = rgb2gray(img)
    # Add a color channel axis to make it compatible with the original shape
    grayscale_img = np.expand_dims(grayscale_img, axis=-1)
    return grayscale_img


def getPredictions(model, threshold, file_path):

    test_data_generator = ImageDataGenerator(rescale=1./255,
                                             preprocessing_function=grayscale_conversion  # Apply grayscale conversion
                                             )
    test_generator = test_data_generator.flow_from_directory(
        "./data/testing_images",
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False  # Set shuffle to False for test data
    )

    y_true = test_generator.classes

    y_proba = model.predict(test_generator)
    y_pred = (y_proba > 0.5).astype(int)

    class_indices = test_generator.class_indices

    return y_true, y_pred, test_generator


#
# Visualizations
# Helper functions that will be called to perform visualizations of model evaluation scores
#

def visualize_predictions_confusionMatrix(title, true_labels, predicted_labels):
    # Plot confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))

    # locs, labels = plt.yticks()
    # plt.yticks([0, 1], ["happy", "neutral"])

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[
                'happy', 'neutral'], yticklabels=[
                'happy', 'neutral'])
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


def analyzePredictions(y_true, y_pred, test_generator):

    visualize_predictions_confusionMatrix("test data", y_true, y_pred)
    visualize_predictions_ROC("test data", y_true, y_pred)
    visualize_predictions_PrecisionRecall("test data", y_true, y_pred)
    # Assuming y_true and y_pred are your true labels and predicted labels respectively

    print(classification_report(y_true, y_pred, zero_division=1))

    # Calculate performance metrics
    fpr, tpr, thresholds = roc_curve(test_generator.classes, y_pred)
    precision = precision_score(test_generator.classes, y_pred)
    recall = recall_score(test_generator.classes, y_pred)
    f1 = f1_score(test_generator.classes, y_pred)
    roc_auc = roc_auc_score(test_generator.classes, y_pred)
    conf_matrix = confusion_matrix(test_generator.classes, y_pred)

    # Print performance metrics
    print("False Positive Rate:", fpr)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC AUC Score:", roc_auc)
    print("Confusion Matrix:")
    print(conf_matrix)


def savePredictions(to_predict_generator):
    output_folder_happy = ".\\data\\test_results\\happy\\"
    output_folder_neutral = ".\\data\\test_results\\neutral\\"
    output_file_path = ".\\data\\test_results\\"

    # Ensure that the output folder exists
    os.makedirs(output_folder_happy, exist_ok=True)
    os.makedirs(output_folder_neutral, exist_ok=True)

    # Iterate over the generator to process and save each image
    for i, (images, _) in enumerate(to_predict_generator):
        for j, image in enumerate(images):
            # Generate a file name for the image

            index = i * to_predict_generator.batch_size + j
            original_file_path = to_predict_generator.filepaths[index]
            # Split the file path into directory and filename with extension
            directory, filename_with_extension = os.path.split(
                original_file_path)

            class_indices = to_predict_generator.class_indices
            subfolder = "neutral"
            if class_indices["happy"] == y_pred[index]:
                subfolder = "happy"
            # subfolder = y_pred[j]
            # Save the image to the output directory
            image_path = os.path.join(
                output_file_path + subfolder + "\\", filename_with_extension)
            # Convert back to uint8 before saving
            Image.fromarray((image * 255).astype(np.uint8)).save(image_path)
        # Stop iteration after processing all batches
        if i + 1 == len(to_predict_generator):
            break

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
    model = load_model(".\\champion_model_drop_0_05.keras")

    y_true, y_pred, to_predict_generator = getPredictions(
        model, 0.6, file_path)
    analyzePredictions(y_true, y_pred, to_predict_generator)

    savePredictions(to_predict_generator)
