# Import necessary libraries
import argparse
import functools
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
# import io
import tensorflow as tf

# Suppress TensorFlow warnings to avoid clutter
tf.get_logger().setLevel('ERROR')

# Load the pre-trained character recognition model
model = load_model('./saved_models/model_char_recognition.keras')

# Define the median height and width for character images
median_height = 100
median_width = 75

# Define the class labels for the character recognition (digits and uppercase letters)
class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                'U', 'V', 'W', 'X', 'Y', 'Z']

# Function to preprocess and predict on a single image
def predict_character(image_array):
    # Convert single-channel image to 3-channel image
    image_array = np.repeat(image_array[..., np.newaxis], 3, axis=-1)
    # Predict the character using the model
    prediction = model.predict(image_array)
    # Get the index of the highest probability class
    predicted_class = np.argmax(prediction, axis=-1)
    # Map the index to the corresponding label
    predicted_label = [class_labels[idx] for idx in predicted_class]
    return predicted_label

# Function to process an image and predict characters
def predict_characters(image_data):
    # Decode image data from memory buffer
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blurring to reduce noise and thresholding for binarization
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 45, 15)

    # Perform connected components analysis to identify individual components
    _, labels = cv2.connectedComponents(thresh)
    # Create a mask to filter out small or large components based on pixel count
    mask = np.zeros(thresh.shape, dtype="uint8")
    total_pixels = image.shape[0] * image.shape[1]
    lower = total_pixels // 200  # Lower bound for pixel count
    upper = total_pixels // 20   # Upper bound for pixel count

    # Filter components based on size
    for (i, label) in enumerate(np.unique(labels)):
        if label == 0:  # Skip the background
            continue
        # Create a mask for the current component
        label_mask = np.zeros(thresh.shape, dtype="uint8")
        label_mask[labels == label] = 255
        # Count the number of pixels in the component
        num_pixels = cv2.countNonZero(label_mask)
        # If the component size is within bounds, add it to the mask
        if num_pixels > lower and num_pixels < upper:
            mask = cv2.add(mask, label_mask)

    # Find contours of the filtered components
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get bounding boxes for each contour
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    # Sort bounding boxes from left to right and top to bottom
    def compare(rect1, rect2):
        # Sort primarily by vertical position, then by horizontal position
        if abs(rect1[1] - rect2[1]) > 10:
            return rect1[1] - rect2[1]
        else:
            return rect1[0] - rect2[0]
    boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare))

    # Preprocess each character and predict
    segmented_characters = []
    for box in boundingBoxes:
        # Extract character image from bounding box
        x, y, w, h = box
        char_img = gray[y:y+h, x:x+w]
        # Resize to median dimensions
        char_img = cv2.resize(char_img, (median_width, median_height))
        # Normalize pixel values
        char_img = np.array(char_img) / 255.0
        # Add a batch dimension
        char_img = np.expand_dims(char_img, axis=0)
        # Append to list of character images
        segmented_characters.append(char_img)

    # Concatenate all character images along the batch dimension
    segmented_characters = np.concatenate(segmented_characters, axis=0)
    # Predict labels for all characters
    predicted_labels = predict_character(segmented_characters)

    # Combine predicted labels into a single string
    license_plate_text = "".join(predicted_labels)
    return license_plate_text
