import matplotlib.pyplot as plt
import numpy as np
import os
import analysis.data_comparison.galacticusAutoAnalysis as analysis
import h5py
import pandas as pd
from latex_plot_style import set_latex_style

set_latex_style()

outputFnameSuffix = "_fullCalibration"
Nsamp = 10 # number of draws from posterior
percentileRange = 0.68


###### LOAD DATA FOR MSTAR-MHALO STUFF
with h5py.File("../galacticusRuns/Low-z_High-z_Sizes/maximumPosterior/romanEPS_massFunction.hdf5") as f:
    maxL = analysis.plot_data(f['analyses/stellarHaloMassRelationLeauthaud2012z1'])
    maxL_highz = analysis.plot_data(f['analyses/stellarHaloMassRelationLeauthaud2012z3'])

posteriorSamplesData = np.zeros((Nsamp, len(maxL.xData)))
posteriorSamplesData_highz = np.zeros((Nsamp, len(maxL_highz.xData)))
for i in np.arange(Nsamp):
    fname = f"/home/arobertson/Galacticus/paperRepositories/CalibratingGalacticus-Paper1/galacticusRuns/Low-z_High-z_Sizes/posteriorDraws/massFunction_{i}/romanEPS_massFunction.hdf5"
    with h5py.File(fname) as ff:
        posteriorSamplesData[i] = analysis.plot_data(ff['analyses/stellarHaloMassRelationLeauthaud2012z1']).yData
        posteriorSamplesData_highz[i] = analysis.plot_data(ff['analyses/stellarHaloMassRelationLeauthaud2012z3']).yData

calibrationMasses = np.array([3.36e11,9.93e11,2.94e12])
massFac = 1.1

# Plot Mhalo - Mstar (z=0.3)
fig, ax = plt.subplots(figsize=(6,4))
# low-z
plt.fill_between(maxL.xData, 10**np.percentile(posteriorSamplesData,50*(1-percentileRange),axis=0), 10**np.percentile(posteriorSamplesData,50*(1+percentileRange),axis=0), color='purple', alpha=0.3)
plt.plot(maxL.xData, 10**maxL.yTarget, lw=3, label='Leauthaud et al. 2012', color='k')
plt.plot(maxL.xData, 10**maxL.yData, lw=3, label='Galacticus', color='purple')


plt.xscale('log')
plt.yscale('log')
for i in np.arange(3):
    plt.axvspan(calibrationMasses[i]/massFac, calibrationMasses[i]*massFac, color='k', alpha=0.3)
plt.xlabel(r"$M_\mathrm{200m} \, / \, \mathrm{M_\odot}$")
plt.ylabel(r"$M_\star \, / \, \mathrm{M_\odot}$")
plt.xlim(1.3e11,1e14)
plt.ylim(8e8,1e12)
plt.legend(frameon=False)
ax.text(
    0.9, 0.96,          # X, Y in axis fraction (top-right corner)
    r'$0.22 < z < 0.48$',    # Text to display
    transform=ax.transAxes,
    ha='right',          # Horizontal alignment
    va='top', fontsize=18   
)
plt.tight_layout()
plt.savefig(f'paperPlotFigures/MstarMhalo{outputFnameSuffix}.pdf')


# Plot Mhalo - Mstar (z=0.9)
calibrationMasses_highz = np.array([4.1e11,9.8e11,2.4e12])
fig, ax = plt.subplots(figsize=(6,4))
# high-z
plt.fill_between(maxL_highz.xData, 10**np.percentile(posteriorSamplesData_highz,50*(1-percentileRange),axis=0), 10**np.percentile(posteriorSamplesData_highz,50*(1+percentileRange),axis=0), color='purple', alpha=0.3)
plt.plot(maxL_highz.xData, 10**maxL_highz.yTarget, lw=3, label='Leauthaud et al. 2012',color='k')
plt.plot(maxL_highz.xData, 10**maxL_highz.yData, lw=3, label='Galacticus', color='purple')



plt.xscale('log')
plt.yscale('log')
for i in np.arange(3):
    plt.axvspan(calibrationMasses_highz[i]/massFac, calibrationMasses_highz[i]*massFac, color='k', alpha=0.3)
plt.xlabel(r"$M_\mathrm{200m} \, / \, \mathrm{M_\odot}$")
plt.ylabel(r"$M_\star \, / \, \mathrm{M_\odot}$")
plt.xlim(4.1e11/1.15,4e13)
plt.ylim(2e9,5e11)
plt.legend(frameon=False)
ax.text(
    0.85, 0.96,          # X, Y in axis fraction (top-right corner)
    r'$0.74 < z < 1$',    # Text to display
    transform=ax.transAxes,
    ha='right',          # Horizontal alignment
    va='top', fontsize=18             
)
plt.tight_layout()
plt.savefig(f'paperPlotFigures/MstarMhalo_highz{outputFnameSuffix}.pdf')

