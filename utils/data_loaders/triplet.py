import tensorflow as tf
import numpy as np

def preprocess_images(images):
    """Normalize images to have a value between 0 and 1."""
    return images.astype("float32") / 255.0

def create_triplets(x, digit_indices, num_triplets):
    """Creates a list of triplets with randomness."""
    triplets = []
    for _ in range(num_triplets):
        # Randomly choose anchor, positive, and negative classes
        anchor_digit = np.random.randint(0, 10)
        negative_digit = (anchor_digit + np.random.randint(1, 10)) % 10
        positive_index = np.random.choice(digit_indices[anchor_digit], 2, replace=False)
        negative_index = np.random.choice(digit_indices[negative_digit], 1)
        
        anchor = x[positive_index[0]]
        positive = x[positive_index[1]]
        negative = x[negative_index[0]]
        
        triplets.append([anchor, positive, negative])
    return np.array(triplets)

def prepare_mnist_data(num_triplets=10000):
    """Load the MNIST dataset and prepare triplets with randomness."""
    # Load the MNIST dataset
    (train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()

    # Preprocess the images
    train_images = preprocess_images(train_images)

    # Create digit indices
    digit_indices = [np.where(train_labels == i)[0] for i in range(10)]

    # Creating triplets
    train_triplets = create_triplets(train_images, digit_indices, num_triplets)

    return train_triplets

# # Specify the number of triplets you want to generate
# num_triplets = 10000
# train_triplets = prepare_mnist_data(num_triplets)

