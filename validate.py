from keras.models import load_model
from dataset import generate_dataset


def main():
    # Generate validation data
    X_val, Y_val = generate_dataset(2000)

    # Load the trained model
    model = load_model('best_pixel_model.h5')

    # Evaluate the model on the validation data
    loss, mae = model.evaluate(X_val, Y_val)
    print(f"Validation Loss: {loss}")
    print(f"Validation MAE: {mae}")


if __name__ == '__main__':
    main()


"""
We have achieved a validation loss of 0.0094 and vaidation MAE of 0.048. 
The validation MAE of 0.048 for predicting the coordinates of a white pixel in a 50x50 image can be considered quite good.
This level of MAE suggests that, on average, the predicted coordinates are less than one-twentieth of a pixel away from the actual coordinates.
"""