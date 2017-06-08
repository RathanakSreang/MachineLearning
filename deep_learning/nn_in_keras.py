# Step:
# 1. Load the dataset(Pima indians onset of diabetes) using python or pandas
# 2. Define neural network model and compile it
# 3. Fit the model to the dataset
# 4. Estimate the performance of the model on unseen data

# dataset url: https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data

# create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")

#split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

#create model
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

#complie model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit the model
model.fit(X,Y, epochs=150, batch_size=10)

#evaluate the model
scores = model.evaluate(X,Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

