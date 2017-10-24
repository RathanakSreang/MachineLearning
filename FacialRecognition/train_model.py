import cv2
import os
import numpy as np

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# load and prepare data
people = ['Cristiano_Ronaldo', 'Jackie_Chan', 'Lionel_Messi', 'Rathanak']
num_classes = 4
img_data_list = []
labels = []
valid_images = [".jpg",".gif",".png"]

for index, person in enumerate(people):
  print(index)
  dir_path = 'images/' + person
  for img_path in os.listdir(dir_path):
    name, ext = os.path.splitext(img_path)
    if ext.lower() not in valid_images:
        continue

    img_data = cv2.imread(dir_path + '/' + img_path)
    # convert image to gray
    img_data=cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    img_data_list.append(img_data)
    labels.append(index)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')

labels = np.array(labels ,dtype='int64')
# scale down(so easy to work with)
img_data /= 255.0
img_data= np.expand_dims(img_data, axis=4)
# print (img_data.shape)
# print (img_data.shape[0])
# print(img_data.shape)
# print(labels.shape)

# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)
# print(Y)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# print(y_train.shape)
# print(y_test.shape)

# Defining the model
input_shape=img_data[0].shape
print(input_shape)

model = Sequential()
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# load model: https://keras.io/models/about-keras-models/
# predict

# refference
# https://keras.io/getting-started/functional-api-guide/
# https://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/
# https://aboveintelligent.com/face-recognition-with-keras-and-opencv-2baf2a83b799
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# https://github.com/fchollet/keras/tree/master/examples
# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.expand_dims.html
