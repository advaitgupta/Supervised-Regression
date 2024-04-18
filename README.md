# Supervised-Regression

This project develops a Convolutional Neural Network (CNN) to accurately predict the coordinates (x, y) of a white pixel in a 50x50 pixel grayscale image, where all other pixels are black. The CNN is trained using TensorFlow/Keras and a custom dataset generator.

## Prerequisites

Before running this project, ensure you have the following installed:
- Python (version 3.6 or newer)
- pip (Python Package Installer)

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/advaitgupta/Supervised-Regression.git
cd Supervised-Regression
```

## Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running Locally

To train the model, navigate to the project directory and execute:

```bash
python train.py
```

To validate the model, navigate to the project directory and execute:

```bash
python validate.py
```

## Running in a Docker Container

Ensure Docker is installed on your system, then build and run the project in a Docker container using Dockerfile

Create a Dockerfile in your project directory with the following content:

```bash
# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /usr/src/app

# Copy the local directory contents to the container
COPY . .

# Install any necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Default command to run on container start
CMD ["python", "train.py"]
```

Build the Docker image and run the container:

```bash
docker build -t pixel-coordinate-predictor .
docker run -it pixel-coordinate-predictor
```

## Project Structure

* train.py: Script to train the CNN model.
* validate.py: Script to evaluate the trained model.
* dataset.py: Script to generate the dataset used for training.
* requirements.txt: Lists dependencies required to run the project.




