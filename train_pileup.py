# root://fax.mwt2.org:1094//atlas/rucio/user.gstark:user.gstark.11630630.OUTPUT._000001.root
import root_numpy as rnp
import numpy as np
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(True)

# input file
filename = 'input.root'
# tree to use for gTowerEt data
treename = 'mytree'
# number of gTowers total
n_gTowers = 1184

maps = rnp.root2array(filename, treename='maps', branches=[('gTowerEta',0,n_gTowers), ('gTowerPhi',0,n_gTowers)], start=0, stop=1)
# we want to mask gTower centers <= 2.4 (this excludes the 0.1x0.2 slice at |eta|~=2.45)
eta_mask = np.fabs(maps['gTowerEta'][0]) <= 2.4

# curious about what eta/phi is being used?
# maps['gTowerEta'][:,eta_mask].reshape(-1, 32)
# maps['gTowerPhi'][:,eta_mask].reshape(-1, 32)

# towers are a 1D vector of something like [(e1,p1), (e1,p2), .... , (e1,pM), (e2, p1), .... , .... (eN, pM)]
#  where e=eta, p=phi
data = rnp.root2array(filename, treename='mytree', branches=['eventNumber', 'gTowerEt', 'Kt4EMTopoEventShape_Density'], start=0, stop=1)

data = rnp.root2array(filename, treename=treename, branches=['eventNumber', ('gTowerEt',0,n_gTowers), 'Kt4EMTopoEventShape_Density'])

# 24 eta rows and 32 phi columns
gTowerEt = data['gTowerEt'][:, np.where(eta_mask)].reshape(-1, 24, 32)
offline_rho = data['Kt4EMTopoEventShape_Density']

''' Start training '''
N_max = 200
N_train = 150

Train_Sample=gTowerEt[:N_train]
Test_Sample=gTowerEt[N_train:N_max]

X_Train=gTowerEt[:N_train]
y_Train=offline_rho[:N_train]

X_Test=gTowerEt[:N_train]
y_Test=offline_rho[N_train:N_max]

import os
os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=cuda,force_device=True"

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(12, input_dim=X_Train.shape[1], init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history=model.fit(X_Train, y_Train, validation_data=(X_Test,y_Test), nb_epoch=10, batch_size=2048)

print history.history

loss_history=history.history["loss"]

print loss_history
