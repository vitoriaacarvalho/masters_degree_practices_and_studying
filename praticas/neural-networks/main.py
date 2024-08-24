import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([ # this means a sequence of layers
    keras.layers.Flatten(input_shape=(28, 28)), # our input layer
    keras.layers.Dense(128, activation="relu"), # flattening the data (hidden layer)
    keras.layers.Dense(10, activation="softmax") # output layer
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5)
# epoch = how many times the model is gonna see this information, how many times you are gonna see the same image
# this is a way to increase hopefully the accuracy of our model

test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print("Tested acc: ", test_accuracy)
