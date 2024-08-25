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

# predict() gives us a group of predictions, so its expecting us to pass a bunch of diff
# things and it predicts them using the model.
prediction = model.predict(test_images)


# print(prediction[0]) --> The output looks like this for one of the indexes:
# [4.74495551e-04 1.69124974e-06 1.11994186e-05 1.38690257e-06
#  1.15848234e-04 7.62295723e-02 3.95751733e-04 4.57947515e-02
#  6.75175761e-05 8.76907885e-01]
#  These are all the different probabilities our network has predicted

# !!!We are gonna take the highest number on this array and consider it the predicted value!!!
# using np.argmax([])
# print(class_names[np.argmax(prediction[0])])


# How can we validate if this is actually working?
# We can show the input and then show what the predicted value is, in that way, the programmer can validate that

for i in range(8):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()
