import h5py
filename = 'file.hdf5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

# Get the data
data = list(f[a_group_key])




#!/usr/bin/env python
import h5py

# Create random data
import numpy as np
data_matrix = np.random.uniform(-1, 1, size=(10, 3))

# Write data to HDF5
data_file = h5py.File('file.hdf5', 'w')
data_file.create_dataset('group_name', data=data_matrix)
data_file.close()
