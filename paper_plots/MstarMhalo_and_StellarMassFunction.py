import matplotlib.pyplot as plt
import numpy as np
import os
import analysis.data_comparison.galacticusAutoAnalysis as analysis
import h5py
import pandas as pd
from latex_plot_style import set_latex_style

set_latex_style()

Nsamp = 10 # number of draws from posterior
percentileRange = 0.68


###### LOAD DATA FOR MSTAR-MHALO STUFF (uses Galactcius run with a single output time equal to the one used in the MCMC likelihood evaluations for a more direct comparison)
with h5py.File("../galacticusRuns/Low-z/maximumPosterior_matchSingleLikelihood/romanEPS_massFunction.hdf5") as f:
    maxL = analysis.plot_data(f['analyses/stellarHaloMassRelationLeauthaud2012z1'])

posteriorSamplesData = np.zeros((Nsamp, len(maxL.xData)))
for i in np.arange(Nsamp):
    fname = f"../galacticusRuns/Low-z/posteriorDraws_matchSingleLikelihood/massFunction_{i}/romanEPS_massFunction.hdf5"
    with h5py.File(fname) as ff:
        mf = analysis.plot_data(ff['analyses/stellarHaloMassRelationLeauthaud2012z1'])
        posteriorSamplesData[i] = mf.yData

calibrationMasses = np.array([3.36e11,9.93e11,2.94e12])
massFac = 1.1

# Plot Mhalo - Mstar
fig, ax = plt.subplots(figsize=(6,4))
plt.fill_between(maxL.xData, 10**np.percentile(posteriorSamplesData,50*(1-percentileRange),axis=0), 10**np.percentile(posteriorSamplesData,50*(1+percentileRange),axis=0), color='purple', alpha=0.3)
plt.plot(maxL.xData, 10**maxL.yTarget, lw=3, label='Leauthaud et al. 2012', color='k')
plt.plot(maxL.xData, 10**maxL.yData, lw=3, label='Galacticus', color='purple')
plt.xscale('log')
plt.yscale('log')
for i in np.arange(3):
    plt.axvspan(calibrationMasses[i]/massFac, calibrationMasses[i]*massFac, color='k', alpha=0.3)
    #plt.axvline(calibrationMasses[i], color='k', ls='--')
plt.xlabel(r"$M_\mathrm{200m} \, / \, \mathrm{M_\odot}$")
plt.ylabel(r"$M_\star \, / \, \mathrm{M_\odot}$")
plt.xlim(1.3e11,1e14)
plt.ylim(8e8,1e12)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(f'paperPlotFigures/MstarMhalo.pdf')


###### LOAD DATA FOR STELLAR MASS FUNCTION STUFF (uses a normal range of output times)
with h5py.File("../galacticusRuns/Low-z/maximumPosterior/romanEPS_massFunction.hdf5") as f:
    SMFmaxL = analysis.plot_data(f['analyses/massFunctionStellarBernardi2013SDSS'])
    SMFmaxL_LiWhite = analysis.plot_data(f['analyses/massFunctionStellarLiWhite2009SDSS'])

posteriorSamplesDataSMF = np.zeros((Nsamp, len(SMFmaxL.xData)))
for i in np.arange(Nsamp):
    fname = f"../galacticusRuns/Low-z/posteriorDraws/massFunction_{i}/romanEPS_massFunction.hdf5"
    with h5py.File(fname) as ff:
        mf = analysis.plot_data(ff['analyses/massFunctionStellarBernardi2013SDSS'])
        posteriorSamplesDataSMF[i] = mf.yData * np.log(10)

# Plot stellar mass function
fig, ax = plt.subplots(figsize=(6,4))
plt.fill_between(SMFmaxL.xData, np.percentile(posteriorSamplesDataSMF,50*(1-percentileRange),axis=0), np.percentile(posteriorSamplesDataSMF,50*(1+percentileRange),axis=0), color='purple', alpha=0.3)
plt.plot(SMFmaxL.xData, SMFmaxL.yTarget*np.log(10), lw=3, color='k', label='Bernardi et al. 2013',zorder=1e3, ls='-')
plt.plot(SMFmaxL_LiWhite.xData, SMFmaxL_LiWhite.yTarget*np.log(10), color='k', label=r'Li \& White 2009',zorder=1e3, ls='--', lw=3)
plt.plot(SMFmaxL.xData, SMFmaxL.yData*np.log(10), lw=3, label='Galacticus', color='purple')
# Appearance stuff
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$M_\star \, / \, \mathrm{M_\odot}$")
plt.ylabel(r"$\Phi \, / \, \mathrm{Mpc}^{-3} \, \mathrm{dex}^{-1}$")
plt.xlim(1e9,2e12)
plt.ylim(2e-6,2e-1)
plt.legend(frameon=False, loc='lower left')
plt.tight_layout()
plt.savefig(f'paperPlotFigures/stellarMassFunction.pdf')