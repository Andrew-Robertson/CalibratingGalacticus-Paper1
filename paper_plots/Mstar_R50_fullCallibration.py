import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import pandas as pd
from latex_plot_style import set_latex_style
from analysis.inference import singleLikelihoods as sl
import analysis.data_comparison.galacticusAutoAnalysis as analysis

set_latex_style()


###### LOAD DATA
with h5py.File("../galacticusRuns/Low-z_High-z_Sizes/maximumPosterior/romanEPS_massFunction.hdf5") as f:
    nd = analysis.hdf5_group_to_dict(f['Outputs/Output16/nodeData'])

central = nd['nodeIsIsolated'][:].astype('bool')
r50 = 1e3*nd['radiusHalfMassStellar'][central]
Mstar = nd['diskMassStellar'][central]+nd['spheroidMassStellar'][central]
BT = nd['spheroidMassStellar'][central]/Mstar
earlyType = (BT>0.5)
lateType = ~earlyType

Dutton11 = pd.read_csv("paperPlotData/Dutton2011_Fig1_ThisPaper_16-50-84.csv")

Ms = np.geomspace(3e8,1e12,100)

fig, ax = plt.subplots(figsize=(6,4))
low, med, high = sl.mean_with_errors(Ms, sl.shen2003_meanLateTypeRadius(Ms), sl.shen2003_sigmaLogR(Ms))
ax.fill_between(Ms, low, high, alpha=0.5, color=sl.coloursShen['disk'])
ax.plot(Ms, sl.shen2003_meanLateTypeRadius(Ms), label='SDSS late-type (Shen+ 2003)', color=sl.coloursShen['disk'])
low, med, high =sl. mean_with_errors(Ms, sl.shen2003_meanEarlyTypeRadius(Ms), sl.shen2003_sigmaLogR(Ms))
ax.fill_between(Ms, low, high, alpha=0.5, color=sl.coloursShen['spheroid'])
ax.plot(Ms, sl.shen2003_meanEarlyTypeRadius(Ms), label='SDSS early-type (Shen+ 2003)', color=sl.coloursShen['spheroid'])
#ax.fill_between(10**Dutton11['logMstar'], 10**Dutton11['logR50_16'], 10**Dutton11['logR50_84'], color='k', alpha=0.3)
ax.plot(10**Dutton11['logMstar'], 10**Dutton11['logR50_50'], label='SDSS disk-dominated (Dutton+ 2011)', color='k')
ax.plot(10**Dutton11['logMstar'], 10**Dutton11['logR50_16'], color='k', ls='--')
ax.plot(10**Dutton11['logMstar'], 10**Dutton11['logR50_84'], color='k', ls='--')
ax.scatter(Mstar[lateType], r50[lateType], label='Galacticus disk-dominated', color=sl.coloursShen['disk'], alpha=0.6, s=0.5*((np.log10(Mstar[lateType]/1e8))**2+2), edgecolors='none', rasterized=True)
ax.scatter(Mstar[earlyType], r50[earlyType], label='Galacticus bulge-dominated', color=sl.coloursShen['spheroid'], alpha=0.6, s=0.5 * 9, edgecolors='none', rasterized=True)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$M_\star \, / \, \mathrm{M_\odot}$')
ax.set_ylabel(r'$R_\star^{50} \, / \, \mathrm{kpc}$')
ax.set_xlim(1e9, 1e12)
ax.set_ylim(0.17, 350)
ax.set_yticks([1,10,100])
ax.set_yticklabels([1,10,100])
legend = ax.legend(loc='upper left', frameon=False, fontsize=10)
plt.tight_layout()

fig.savefig('paperPlotFigures/Mstar_R50_fullCallibration.pdf', dpi=300)