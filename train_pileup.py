# root://fax.mwt2.org:1094//atlas/rucio/user.gstark:user.gstark.11630630.OUTPUT._000001.root
import root_numpy as rnp
import numpy as np
np.random.seed(7)
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(True)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

import argparse
parser = argparse.ArgumentParser(description='Run ML model')
parser.add_argument('file', type=str, help='File to process', default='input.root')

args = parser.parse_args()

def baseline_model():
  model = Sequential()
  model.add(Dense(32*24, input_dim=32*24, activation='relu', kernel_initializer='normal'))
  model.add(Dense(256, activation='relu', kernel_initializer='normal'))
  model.add(Dense(1, activation='relu', kernel_initializer='normal'))
  model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
  """
  # ValueError: Input 0 is incompatible with layer conv2d_1: expected ndim=4, found ndim=3
  #	gotta use (24, 32, 1) and not (24, 32) -- WHY?
  model.add(Conv2D(64, (5, 5), input_shape=(24, 32, 1), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.2))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
  """
  model.summary()
  return model

def get_data(filename, treename='mytree', n_gTowers=1184, eta_max=2.4, flattened_input=True, normalize=True):
  maps = rnp.root2array(filename, treename='maps', branches=[('gTowerEta',0,n_gTowers), ('gTowerPhi',0,n_gTowers)], start=0, stop=1)
  # we want to mask gTower centers <= 2.4 (this excludes the 0.1x0.2 slice at |eta|~=2.45)
  eta_mask = np.fabs(maps['gTowerEta'][0]) <= eta_max

  # curious about what eta/phi is being used?
  # maps['gTowerEta'][:,eta_mask].reshape(-1, 32)
  # maps['gTowerPhi'][:,eta_mask].reshape(-1, 32)

  # towers are a 1D vector of something like [(e1,p1), (e1,p2), .... , (e1,pM), (e2, p1), .... , .... (eN, pM)]
  #  where e=eta, p=phi
  data = rnp.root2array(filename, treename=treename, branches=['eventNumber', ('gTowerEt',0,n_gTowers), 'Kt4EMTopoEventShape_Density'])

  # 24 eta rows and 32 phi columns
  gTowerEt = data['gTowerEt'][:, np.where(eta_mask)].reshape(-1, 24, 32, 1)
  if flattened_input:
    gTowerEt = gTowerEt.reshape(-1, 24*32)
  offline_rho = data['Kt4EMTopoEventShape_Density']

  if normalize:
    # normalize rho and save min/max to convert back again if needed
    #normalizations = (offline_rho.max(), offline_rho.min())
    offline_rho = (offline_rho - offline_rho.min())/(offline_rho.max() - offline_rho.min())
    gTowerEt = (gTowerEt - gTowerEt.min())/(gTowerEt.max() - gTowerEt.min())

  return (gTowerEt, offline_rho)

gTowerEt, offline_rho = get_data(args.file)
model = baseline_model()

history=model.fit(gTowerEt, offline_rho, epochs=100, batch_size=32, verbose=2)

''' Start training '''
"""
N_max = 200
N_train = 150

gTowerEt, offline_rho = get_data('input.root')

X_Train=gTowerEt[:N_train]
y_Train=offline_rho[:N_train]

X_Test=gTowerEt[N_train:N_max]
y_Test=offline_rho[N_train:N_max]

import os
os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=cuda,force_device=True"
model = baseline_model()

history=model.fit(X_Train, y_Train, validation_data=(X_Test,y_Test), epochs=100, batch_size=32, verbose=2)

loss_history=history.history["loss"]
#model.save('model.hd5')
"""
