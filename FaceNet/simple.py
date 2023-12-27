# pip install tensorflow tensorflow-addons

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained FaceNet model
face_net_model = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3))

# Function to preprocess an image for FaceNet
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((160, 160))
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to get FaceNet embeddings for an image
def get_face_embedding(image_path):
    img_array = preprocess_image(image_path)
    embedding = face_net_model.predict(img_array)
    return embedding.flatten()

# Example usage
image_path_1 = 'path/to/your/image1.jpg'
image_path_2 = 'path/to/your/image2.jpg'

embedding_1 = get_face_embedding(image_path_1)
embedding_2 = get_face_embedding(image_path_2)

# Calculate the cosine similarity between embeddings
cosine_similarity = tfa.losses.metric_learning.cosine_similarity(embedding_1, embedding_2)
print(f'Cosine Similarity: {cosine_similarity.numpy()}')

# Visualize the images
plt.subplot(1, 2, 1)
plt.imshow(Image.open(image_path_1))
plt.title('Image 1')

plt.subplot(1, 2, 2)
plt.imshow(Image.open(image_path_2))
plt.title('Image 2')

plt.show()
