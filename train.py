import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from dataset import generate_dataset


def create_model(input_shape=(50, 50, 1)):
    """
    Creates and compiles a CNN model for predicting pixel coordinates.

    Args:
        input_shape (tuple): Shape of the input images.

    Returns:
        tensorflow.keras.Model: Compiled CNN model.
    """
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(2, activation='linear')  # Predicting x, y coordinates
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model



def main():
    # Generate training data
    X_train, Y_train = generate_dataset(8000)

    # Create and train the model
    model = create_model()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        ModelCheckpoint('best_pixel_model.h5', monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1),
        TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
    ]

    # Fit model
    model.fit(
        X_train, Y_train,
        epochs=50,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    print("Training complete. Best model saved as 'best_pixel_model.h5'.")


if __name__ == '__main__':
    main()


"""
Based on the training, we have achieved a final training loss of 0.0049 and a training MAE (Mean Absolute Error) of 0.0327
We have also achieved a validation loss of 0.00796 and validation MAE of 0.0481
This signifies that the model is not overfitting and performing fairly well.
We have also tested the model on a separate validation dataset generate by the same dataset.py function whose results are mentioned at the end of the validate.py script.
"""