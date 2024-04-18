import numpy as np

def generate_dataset(num_samples=10000, img_size=(50, 50)):
    """
    Generates a dataset of images where each image contains exactly one pixel
    set to 255 (white) and all others are 0 (black).

    Args:
        num_samples (int): Number of samples to generate.
        img_size (tuple): Dimensions of the images (height, width).

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Array of shape (num_samples, img_size[0], img_size[1], 1)
              representing the dataset images.
            - Y (numpy.ndarray): Array of shape (num_samples, 2) representing the
              pixel coordinates for each image.
    """
    X = np.zeros((num_samples, img_size[0], img_size[1], 1), dtype=np.float32)
    Y = np.zeros((num_samples, 2), dtype=np.int32)

    for i in range(num_samples):
        x, y = np.random.randint(0, img_size[0]), np.random.randint(0, img_size[1])
        X[i, x, y, 0] = 255
        Y[i] = [x, y]

    return X, Y


"""
Rationale behind the choice of dataset:
The dataset was specifically designed for the precise task of identifying a single white pixel in a 50x50 grayscale image. 
Each image is minimalistic, featuring one white pixel on a black background, ensuring the dataset directly supports the learning objective. 
This setup simplifies the learning process and covers all possible pixel positions, 
effectively helping the model learn to accurately localize the single differing pixel, 
thus enhancing its robustness and generalization capabilities.
"""
