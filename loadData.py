#
#  Juliane Bruck  - 8297746
#  Assignment #2
#
# Based on https://github.com/cocoxu/SemEval-PIT2015
#
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score


# Set seeds for NumPy and TensorFlow random number generators
np.random.seed(128)
tf.random.set_seed(128)

m_rootFolder = ".\\data\\SMILE PLUS Training Set\\SMILE PLUS Training Set\\"
m_csvData = m_rootFolder + "annotations.csv"

#  true image dimensions are (162, 193) `rows` is not in [96, 128, 160, 192, 224]. Weights for input shape (224, 224)
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


# Define column names for data
m_columns = ["FILENAME", "EMOTION"]


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


# Define a function to convert images to grayscale
def grayscale_conversion(img):
    # Convert the image to grayscale
    grayscale_img = rgb2gray(img)
    # Add a color channel axis to make it compatible with the original shape
    grayscale_img = np.expand_dims(grayscale_img, axis=-1)
    return grayscale_img


def create_smile_detection_model(base_model):
    # Freeze the base_model
    base_model.trainable = False

    # Add custom top layers for smile detection
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)  # Add a dense layer with 128 units
    # Add another dense layer with 64 units
    x = Dense(64, activation='relu')(x)
    # Output layer with sigmoid activation for binary classification
    predictions = Dense(1, activation='sigmoid')(x)

    # This is the model we will train for smile detection
    smile_detection_model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    smile_detection_model.compile(optimizer='adam',
                                  loss='binary_crossentropy',
                                  metrics=['accuracy'])

    return smile_detection_model


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


def createMobileNetV2Model():

    # Load pre-trained VGG16 model
    # base_model = VGG16(weights='imagenet')
    # Load the pre-trained MobileNetV2 model, excluding its top (fully connected) layer
    base_model = MobileNetV2(
        weights='imagenet', include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    # Freeze the base_model
    base_model.trainable = False

    #
    # Add custom layers on top for our specific task such as smile detection
    #

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # Add a dense layer with 128 units
    x = Dense(128, activation='relu')(x)
    # Add another dense layer with 64 units
    x = Dense(64, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)

    # 2 classes: smile and neutral
    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # model.summary()
    return model


def trainMobileNetV2Model(model):
    # Data preprocessing
    train_datagen = getImageDataGenerator()
    # Change 'train_directory' and 'validation_directory' to your dataset's directories
    train_generator = train_datagen.flow_from_directory(
        ".\\data\\training_images",
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=32,
        class_mode='categorical',
        subset='training')

    validation_datagen = getImageDataGenerator()
    validation_generator = validation_datagen.flow_from_directory(
        ".\\data\\validation_images",
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=32,
        class_mode='categorical',
        subset='validation')
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.n // 32,
        epochs=1,  # You can adjust the number of epochs
        validation_data=validation_generator,
        validation_steps=validation_generator.n // 32)
    # Print the training and validation accuracy
    print("Model Accuracy:", history.history['accuracy'][0])
    print("Model loss:", history.history['loss'][0])
    # model.summary()
    return model


def getPredictions(model, threshold):

    prediction_datagen = getImageDataGenerator()

    to_predict_generator = prediction_datagen.flow_from_directory(
        ".\\data\\testing_images",
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
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


#
# Helpers to extract features
#


# Function to preprocess the input image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    img_array = image.img_to_array(img)
    # Expand dimensions to create batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array


def extract_hog_features(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate HOG features
    features, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True)
    # Rescale histogram for better visualization
    hog_image_rescaled = exposure.rescale_intensity(
        hog_image, in_range=(0, 10))
    return features


def extract_lbp_features(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate LBP features
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3),
                           range=(0, n_points + 2))
    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


def extract_cnn_features(image):
    # Load pre-trained VGG16 model
    model = VGG16(weights='imagenet', include_top=False)
    # Preprocess the input image
    img = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    # Extract features from VGG16 model
    features = model.predict(img)
    # Flatten the features
    features = features.flatten()
    return features


# Load and preprocess the image


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    # Resize image to the input size required by VGG16
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)        # Preprocess image
    return img


# Predict smile or not smile


if __name__ == "__main__":

    model = createMobileNetV2Model()
    model = trainMobileNetV2Model(model)

    # Save the trained model
    model.save('smile_detection_model.keras')

    y_true, y_pred, to_predict_generator = getPredictions(model, 0.6)
    analyzePredictions(y_true, y_pred, to_predict_generator)

    # # # Extract HOG features
    # hog_features = extract_hog_features(image)

    # # # Extract LBP features
    # lbp_features = extract_lbp_features(image)

    # # # Extract CNN features
    # cnn_features = extract_cnn_features(image)

    # print("Row {}: Column1={}, Column2={}".format(index, imageFile, emotion))
    # print("HOG features:", hog_features.shape)
    # print("LBP features:", lbp_features.shape)
    # print("CNN features:", cnn_features.shape)
    # Process row data
