import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.losses import mean_squared_error

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

carData_train = pd.read_csv("carOvertakeTrainData.csv", names=["Initial Seperation", "Overtaking Speed MPS", "Oncoming Speed", "Success?"])
carData_train.head()

carData_features = carData_train.copy()
carData_labels = carData_features.pop('Success?')

carData_features = np.array(carData_features)
carData_features

carData_model = tf.keras.Sequential([
    layers.Dense(128),
    layers.Dense(64),
    layers.Dense(1)
])

carData_model.compile(optimizer=tf.optimizers.Adam(), loss= mean_squared_error, metrics=['accuracy'])
history = carData_model.fit(carData_features, carData_labels, epochs=40)
#history = carData_model.fit(carData_features, carData_labels, epochs=20)
print("Finished Training the Model")
#print(history.history)

carData_test = pd.read_csv("testData.csv", names=["Initial Seperation", "Overtaking Speed MPS", "Oncoming Speed", "Success?"])
carData_test.head()

carData_featuresTest = carData_test.copy()
carData_labelsTest = carData_featuresTest.pop('Success?')

#result = {}

result = carData_model.evaluate(carData_featuresTest, carData_labelsTest, verbose=0)
print(result)

print(history.history.keys())
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show(block=True)

