# root://fax.mwt2.org:1094//atlas/rucio/user.gstark:user.gstark.11630630.OUTPUT._000001.root
import numpy as np
np.random.seed(7)
import h5py

import os
os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=cuda,force_device=True"

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.wrappers.scikit_learn import KerasRegressor

import argparse
parser = argparse.ArgumentParser(description='Run ML model')
parser.add_argument('file', type=str, help='File to process', default='input.hdf5')

args = parser.parse_args()

def baseline_model(simple=True):
  model = Sequential()
  if simple:
    model.add(Dense(32*24, input_dim=32*24, activation='relu', kernel_initializer='normal'))
    model.add(Dense(256, activation='relu', kernel_initializer='normal'))
  else:
    # ValueError: Input 0 is incompatible with layer conv2d_1: expected ndim=4, found ndim=3
    #	gotta use (24, 32, 1) and not (24, 32) -- WHY?
    model.add(Conv2D(32, (5, 5), input_shape=(24, 32, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))

  # output is always size 1, add this last layer always
  model.add(Dense(1, activation='linear', kernel_initializer='normal'))
  model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
  model.summary()
  return model

def get_data(filename, flattened_input=True, normalize=True):
  data = h5py.File(filename, 'r')

  # 24 eta rows and 32 phi columns
  gTowerEt = data['gTowerEt'][:].reshape(-1, 24, 32, 1)
  if flattened_input:
    gTowerEt = gTowerEt.reshape(-1, 24*32)
  offlineRho = data['offlineRho'][:]

  if normalize:
    # normalize rho and save min/max to convert back again if needed
    #normalizations = (offlineRho.max(), offlineRho.min())
    offlineRho = (offlineRho - offlineRho.min())/(offlineRho.max() - offlineRho.min())
    gTowerEt = (gTowerEt - gTowerEt.min())/(gTowerEt.max() - gTowerEt.min())

  return (gTowerEt, offlineRho)

# Simple: Dense layers, flattened (N, -1) input
# Non-Simple: Convolution 2D, image (N, 24, 32, 1) input
simple=True
gTowerEt, offlineRho = get_data(args.file, flattened_input=simple)
model = baseline_model(simple)
history=model.fit(gTowerEt, offlineRho, epochs=100, batch_size=1024, verbose=2)

"""use a regressor instead?
clf = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=1024, verbose=2)
history = clf.fit(gTowerEt, offlineRho)
"""

"""training example?
N_max = 200
N_train = 150

X_Train=gTowerEt[:N_train]
y_Train=offlineRho[:N_train]

X_Test=gTowerEt[N_train:N_max]
y_Test=offlineRho[N_train:N_max]

model = baseline_model()

history=model.fit(X_Train, y_Train, validation_data=(X_Test,y_Test), epochs=100, batch_size=32, verbose=2)

loss_history=history.history["loss"]
#model.save('model.hd5')
"""
