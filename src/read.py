import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# Correct path to the dataset
dataset_path = 'E:\\Applications\\Anaconda\\envs\\nvidia_env\\Projects\\Data\\neural-networks-and-deep-learning-master\\data\\mnist.pkl'

# Load the dataset
with open(dataset_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# Extract training data
training_data, validation_data, test_data = data
training_images, training_labels = training_data

# Define the output directory (updated to "Testdata")
output_dir = 'E:\\Applications\\Anaconda\\envs\\nvidia_env\\Projects\\Data\\neural-networks-and-deep-learning-master\\data\\Testdata\\'

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to save individual images
def save_images(images, labels, num_images=5):
    for i in range(num_images):
        img = np.reshape(images[i], (28, 28))
        plt.imshow(img, cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
        # Save the image to a file
        file_path = os.path.join(output_dir, f'image_{i}_label_{labels[i]}.png')
        plt.savefig(file_path)
        plt.close()

# Save the first 5 images in the training dataset
save_images(training_images, training_labels, num_images=5)
