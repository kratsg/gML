import numpy as np
# fix random seed for reproducibility
np.random.seed(7)

N = int(1e4)

mu0 = np.random.normal(0, 1, N)
mu0p5 = np.random.normal(0.5, 1, N)

'''from Ben
model = Sequential()
model.add(Dense(12, input_dim=1, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(mu0, mu0p5, epochs=150, batch_size=10)
'''

from keras.models import Sequential
from keras.layers import Activation, Dense

model = Sequential()
model.add(Dense(12, input_dim=N))
model.add(Activation('relu'))
model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history=model.fit(X_Train, y_Train, validation_data=(X_Test,y_Test), nb_epoch=10, batch_size=2048)

print history.history

loss_history=history.history["loss"]

print loss_history
