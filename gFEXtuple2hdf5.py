import h5py
import root_numpy as rnp
import numpy as np
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(True)

import argparse
parser = argparse.ArgumentParser(description='Convert gTower Et in a gFEX ntuple to HDF5')
parser.add_argument('files', type=str, nargs='+', help='Files to process')
parser.add_argument('--output', type=str, help='Output filename', default='output.hdf5')

args = parser.parse_args()

def get_data(filename, treename='mytree', n_gTowers=1184, eta_max=2.4):
  maps = rnp.root2array(filename, treename='maps', branches=[('gTowerEta',0,n_gTowers), ('gTowerPhi',0,n_gTowers)], start=0, stop=1)
  # we want to mask gTower centers <= 2.4 (this excludes the 0.1x0.2 slice at |eta|~=2.45)
  eta_mask = np.fabs(maps['gTowerEta'][0]) <= eta_max

  # curious about what eta/phi is being used?
  # maps['gTowerEta'][:,eta_mask].reshape(-1, 32)
  # maps['gTowerPhi'][:,eta_mask].reshape(-1, 32)

  # towers are a 1D vector of something like [(e1,p1), (e1,p2), .... , (e1,pM), (e2, p1), .... , .... (eN, pM)]
  #  where e=eta, p=phi
  data = rnp.root2array(filename, treename=treename, branches=[('gTowerEt',0,n_gTowers), 'Kt4EMTopoEventShape_Density'])

  # 24 eta rows and 32 phi columns
  gTowerEt = data['gTowerEt'][:, np.where(eta_mask)].reshape(-1, 24, 32)
  offlineRho = data['Kt4EMTopoEventShape_Density']

  return gTowerEt, offlineRho

output_file = h5py.File(args.output, 'w')
out_gTowerEt = None
out_offlineRho = None

for f in args.files:
  gTowerEt, offlineRho = get_data(f)
  if 'gTowerEt' not in output_file:
    out_gTowerEt = output_file.create_dataset('gTowerEt', data=gTowerEt, maxshape=(None, 24, 32))
    out_offlineRho = output_file.create_dataset('offlineRho', data=offlineRho, maxshape=(None,))
  else:
    for out_dset, new_data in zip([out_gTowerEt, out_offlineRho], [gTowerEt, offlineRho]):
      out_dset.resize(out_dset.shape[0]+len(new_data), axis=0)
      out_dset[-len(new_data):] = new_data
