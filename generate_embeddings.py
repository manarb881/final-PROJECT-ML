import os
import numpy as np
from keras_facenet import FaceNet
from PIL import Image

# Define the folder where the images are stored
DATASET_FOLDER = 'datasets/lfw-deepfunneled'

# Initialize FaceNet model
model = FaceNet().model

def preprocess_image(image_path):
    """Load and preprocess the image."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((160, 160))  # Resize to 160x160
    img_array = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
    return img_array

def image_to_embedding(image_array, model):
    """Convert an image array to its embedding."""
    image_array = np.expand_dims(image_array, axis=0)
    embedding = model.predict(image_array)
    return embedding.flatten()

def generate_embeddings(dataset_folder):
    """Generate embeddings for all images in the dataset."""
    embeddings = []
    labels = []
    person_names = []

    for person_name in os.listdir(dataset_folder):
        person_folder = os.path.join(dataset_folder, person_name)
        if os.path.isdir(person_folder):
            for image_file in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_file)
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_array = preprocess_image(image_path)
                    embedding = image_to_embedding(image_array, model)
                    embeddings.append(embedding)
                    labels.append(person_name)
                    if person_name not in person_names:
                        person_names.append(person_name)

    # Save embeddings and labels
    np.save('/Users/pc/myWork/models/lfw_embeddings.npy', embeddings)
    np.save('/Users/pc/myWork/models/lfw_labels.npy', labels)

    return np.array(embeddings), np.array(labels), person_names

if __name__ == '__main__':
    generate_embeddings(DATASET_FOLDER)

