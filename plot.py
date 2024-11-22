from keras.models import load_model
import matplotlib.pyplot as pl
import numpy as np
from train_pileup import get_data

gTowerEt, offlineRho = get_data('user.gstark.117050.PowhegPythia_P2011C_ttbar.e2176_s3142_s3143_r9589.mu200.v30_compile_OUTPUT.hdf5', flattened_input=False, normalize=True)

gTowerEt_true = gTowerEt[-10000:]
offlineRho_true = offlineRho[-10000:]

model = load_model('model.hd5')
offlineRho_pred = model.predict(gTowerEt_true)

offlineRho_true = offlineRho_true.reshape(-1)
offlineRho_pred = offlineRho_pred.reshape(-1)

"""
fig, ax = pl.subplots(figsize=(10,10))
counts, xedges, yedges, im = ax.hist2d(offlineRho[:-10000].reshape(-1), np.median(gTowerEt[:-10000], axis=1).reshape(-1), bins=50, cmap=pl.get_cmap('Wistia'))
ax.set_xlabel('$\\rho_\mathrm{offline}^\mathrm{true}$')
ax.set_ylabel('median $E_\mathrm{T}^\mathrm{gFEX} [MeV]$')
ax.set_xlim([0.0,1.0])
fig.colorbar(im, ax=ax)
fig.savefig('plot_training.pdf', bbox_inches='tight')
"""

fig, ax = pl.subplots(figsize=(10,10))
counts, xedges, yedges, im = ax.hist2d(offlineRho_true, offlineRho_pred, bins=50, cmap=pl.get_cmap('Wistia'))
ax.set_xlabel('$\\rho_\mathrm{offline}^\mathrm{true}$')
ax.set_ylabel('$\\rho_\mathrm{offline}^\mathrm{predicted}$')
ax.set_xlim([0.0,1.0])
ax.set_ylim([0.0,1.0])
fig.colorbar(im, ax=ax)
fig.savefig('plot_validation.pdf', bbox_inches='tight')
