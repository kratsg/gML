# root://fax.mwt2.org:1094//atlas/rucio/user.gstark:user.gstark.11630630.OUTPUT._000001.root
import numpy as np
np.random.seed(7)
import h5py

import os
# you know you have a GPU and you've verified it works? Uncomment the next line
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization

def baseline_model():
  model = Sequential()
  model.add(Conv2D(32, (3, 3), input_shape=(24, 32, 1), activation='relu'))
  model.add(BatchNormalization())

  # output is always size 1, add this last layer always
  model.add(Flatten())
  model.add(Dense(1, activation='linear', kernel_initializer='normal'))
  model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
  model.summary()
  return model

def get_data(filename, normalize=True):
  data = h5py.File(filename, 'r')

  # 24 eta rows and 32 phi columns
  gTowerEt = data['gTowerEt'][:].reshape(-1, 24, 32, 1)
  offlineRho = data['offlineRho'][:]

  if normalize:
    # normalize rho and save min/max to convert back again if needed
    #normalizations = (offlineRho.max(), offlineRho.min())
    offlineRho = (offlineRho - offlineRho.min())/(offlineRho.max() - offlineRho.min())
    gTowerEt = (gTowerEt - gTowerEt.min())/(gTowerEt.max() - gTowerEt.min())

  return (gTowerEt, offlineRho)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run ML model')
  parser.add_argument('file', type=str, help='File to process', default='input.hdf5')

  args = parser.parse_args()

  # Simple: Dense layers, flattened (N, -1) input
  # Non-Simple: Convolution 2D, image (N, 1, 24, 32) input
  gTowerEt, offlineRho = get_data(args.file)
  model = baseline_model()
  history=model.fit(gTowerEt[:-10000], offlineRho[:-10000], validation_data=(gTowerEt[-10000:], offlineRho[-10000:]), epochs=100, batch_size=512, verbose=2)
  model.save('model.hd5')

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
