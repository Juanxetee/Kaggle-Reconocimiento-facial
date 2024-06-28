import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

# Function to load and preprocess images
def load_images_from_folder(folder, label, img_size=(48, 48)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = load_img(img_path, target_size=img_size, color_mode='grayscale')
        img_array = img_to_array(img)
        img_array /= 255.0  # Normalizing the image
        images.append(img_array)
        labels.append(label)
    return np.array(images), np.array(labels)