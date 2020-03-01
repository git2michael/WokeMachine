import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#data = keras.datasets.fashion_mnist
data = "/Users/ericchang/vgg_face2"

(train_images, train_labels), (test_images, test_labels) = data.load_data()

#different types of categories
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

#print(train_images[7])
#plt.imshow(train_images[7], cmap = plt.cm.binary)
#plt.show()

#creates layers of neural network
model = keras.Sequential([
	keras.layers.Flatten(input_shape = (28,28)),
	keras.layers.Dense(128, activation = "relu"),
	keras.layers.Dense(256, activation = "relu"),
	keras.layers.Dense(512, activation = "relu"),
	keras.layers.Dense(10, activation = "softmax")
	])

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
model.fit(train_images, train_labels, epochs = 5)

#finds percent error for the training
#test_loss, test_acc = model.evaluate(test_images, test_labels)
#print("Tested Acc:", test_acc)

#computer prediction for the different pictures
prediction = model.predict(test_images)
#prediction = model.predict([test_images[7]]) #test for one item

for i in range(5):
	plt.grid(False)
	plt.imshow(test_images[i], cmap = plt.cm.binary)
	plt.xlabel("Actual: "  + class_names[test_labels[i]])
	plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
	plt.show()
#takes largest value of neuron and gives value
#print(class_names[np.argmax(prediction[0])])