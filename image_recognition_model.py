# -*- coding: utf-8 -*-
"""
This script trains a convolutional neural network (CNN) on the CIFAR-10 dataset using TensorFlow and Keras. 
It includes functionality to load an existing model, train a new model if none exists, and save the best-performing model. 
Additionally, it evaluates the trained model on the test dataset and displays a random test image with its predicted label.
The main components of the script are:
- Loading and normalizing the CIFAR-10 dataset.
- Defining the CNN architecture.
- Compiling the model.
- Implementing callbacks for model checkpointing, early stopping, TensorBoard logging, and learning rate scheduling.
- Training the model.
- Saving the model.
- Evaluating the model on the test dataset.
- Displaying a random test image with its predicted label.

@author: Tomas Arzola Röber
"""

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
import matplotlib.pyplot as plt
import numpy as np
import os

# Path to the saved model
saved_model_path = 'cifar.h5'

# Check if a saved model exists
model_exists = os.path.exists(saved_model_path)

# Load the CIFAR-10 dataset and normalize pixel values to the range [0, 1]
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the model architecture
if model_exists:
    # If a saved model exists, load it
    model = load_model(saved_model_path)
    print("Loading previously trained model...")
    # Recompile the loaded model to ensure metrics are correctly set
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
else:
    # If no saved model exists, define a new model
    model = Sequential([
        Conv2D(128, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4),

        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.6),
        Dense(10, activation='softmax')
    ])

    # Compile the newly defined model
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Define callbacks for model checkpointing, early stopping, TensorBoard, and learning rate scheduling
    checkpoint = ModelCheckpoint(saved_model_path, monitor='val_accuracy', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, mode='max')
    log_dir = "logs/CIFAR"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Learning rate scheduler function
    def lr_scheduler(epoch, lr):
        if epoch > 0 and epoch % 5 == 0:
            return lr * 0.5
        return lr

    lr_callback = LearningRateScheduler(lr_scheduler)

    # Train the model
    model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test),
              callbacks=[checkpoint, early_stopping, lr_callback, tensorboard_callback])

    # Save the trained model
    model.save(saved_model_path)
    print(f"Model saved as {saved_model_path}")

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_acc}')

# Function to show a random test image and its prediction
def test_image():
    # Select a random image from the test set
    index = np.random.randint(0, x_test.shape[0])
    image = x_test[index]
    true_label = y_test[index][0]
    # Predict the label for the selected image
    prediction = np.argmax(model.predict(np.expand_dims(image, axis=0)))

    # Display the image with true label and predicted label
    plt.imshow(image)
    plt.title(f'True label: {true_label}, Prediction: {prediction}')
    plt.show()

# Show a random test image and its prediction
test_image()

