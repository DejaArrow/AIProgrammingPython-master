import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.losses import mean_squared_error

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

#Reads the CSV file with the training data and creates headers for the columns.
carData_train = pd.read_csv("carOvertakeTrainData.csv", names=["Initial Seperation", "Overtaking Speed MPS", "Oncoming Speed", "Success?"])
carData_train.head()

#Copies the data into features - the inputs
carData_features = carData_train.copy()

#Pops out the fourth column as the labels - the output
carData_labels = carData_features.pop('Success?')

carData_features = np.array(carData_features)
carData_features

#Sets how many layers and amount of neurons
carData_model = tf.keras.Sequential([
    layers.Dense(128),
    layers.Dense(64),
    layers.Dense(1)
])

#Compiles the model with preferred optimiser and metrics for loss and accuracy.
carData_model.compile(optimizer=tf.optimizers.Adam(), loss= mean_squared_error, metrics=['accuracy'])
history = carData_model.fit(carData_features, carData_labels, epochs=40)
print("Finished Training the Model")

#Read test data CSV
carData_test = pd.read_csv("testData.csv", names=["Initial Seperation", "Overtaking Speed MPS", "Oncoming Speed", "Success?"])
carData_test.head()

carData_featuresTest = carData_test.copy()
carData_labelsTest = carData_featuresTest.pop('Success?')

#Get result of test data and print accuracy metrics to screen
result = carData_model.evaluate(carData_featuresTest, carData_labelsTest, verbose=0)
print(result)

#Plot and print Accuracy graph of training model
print(history.history.keys())
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show(block=True)

