import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import pandas as pd
from latex_plot_style import set_latex_style
from getdist import MCSamples, plots
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from analysis.inference import singleLikelihoods as sl
import analysis.data_comparison.galacticusAutoAnalysis as analysis
from analysis.inference import mcmc

plt.rcParams.update({
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'axes.labelsize': 24,
    'legend.fontsize': 20,
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'axes.unicode_minus': False,
})

results = mcmc.MCMC_results("../galacticusMCMCs/Low-z_High-z_Sizes/run", burnLen=0.1)
results.readFiles()
SHMR = sl.SHMR_likelihood("../galacticusMCMCs/Low-z_High-z_Sizes/singleLikelihood", lowzFname="romanEPS_singleLikelihood_lowz.hdf5", highzFname="romanEPS_singleLikelihood_highz.hdf5")
SHMR.load_data()
sizes = sl.mass_size_likelihood("../galacticusMCMCs/Low-z_High-z_Sizes/singleLikelihood", fname="romanEPS_singleLikelihood_lowz.hdf5")
sizes.load_data()

parameters = results.paramNames
idxs = [parameters.index(p) for p in parameters]
samples = results.combined_df[parameters].to_numpy()[:, idxs]

ranges={'diskExponent':(0.0, 5.0), 'henriquesGamma':(1.0, 25.0), 'henriquesDelta1':(-2.0,4.0), 'henriquesDelta2':(-1.0,5.0), 'BHefficiencyRadioMode':(0.03,1.0), 'spheroidRatioAngularMomentumScaleRadius':(0.1,0.5)}
labels = [r'V_\mathrm{disk} \, / \, \mathrm{km \, s^{-1}}', r'\alpha_\mathrm{disk}', r'\gamma', r'\delta_1', r'\delta_2', r'\epsilon_\mathrm{AGN}', r'R_2 / R_3']


gds_smooth = MCSamples(samples=samples, names=parameters, labels=labels, settings={"smooth_scale_1D": 5.0, "smooth_scale_2D": 3.0}, ranges=ranges)
gds = MCSamples(samples=samples, names=parameters, labels=[f"{p}" for p in parameters])

######## Additional posteriors
lowzSHMR = mcmc.MCMC_results("../galacticusMCMCs/Low-z/run", burnLen=0.1)
lowzSHMR.readFiles()
lowzSHMR_smooth = MCSamples(samples=lowzSHMR.combined_df[parameters].to_numpy()[:, idxs], names=parameters, labels=labels, settings={"smooth_scale_1D": 5.0, "smooth_scale_2D": 3.0}, ranges=ranges)

highzSHMR = mcmc.MCMC_results("../galacticusMCMCs/High-z/run", burnLen=0.1)
highzSHMR.readFiles()
highzSHMR_smooth = MCSamples(samples=highzSHMR.combined_df[parameters].to_numpy()[:, idxs], names=parameters, labels=labels, settings={"smooth_scale_1D": 5.0, "smooth_scale_2D": 3.0}, ranges=ranges)




############
colors = ['gray', 'navy']

g = plots.get_subplot_plotter()
g.settings.axes_fontsize = 30
g.settings.axes_labelsize = 30
#g.triangle_plot([gds, gds_smooth], filled=True)
g.settings.line_labels =False
g.triangle_plot([lowzSHMR_smooth, gds_smooth,], filled=True, contour_colors=colors)
fig = g.fig  # Get the figure object
# legend (not sure how to move and change fontsize of auto-produce one)
  # Replace with your own color hex or names
labels = [r'low-$z$ SHMR', r'full calibration']
# Create legend handles (filled color blocks)
handles = [Patch(color=color, label=label) for color, label in zip(colors, labels)]
# Add legend to the figure at an arbitrary location (e.g., top right corner)
fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.5, 0.98), frameon=False)

                
'''
for ax in g.subplots.flatten():
    if ax is not None:
        if ax.get_xlabel() == '$R_2 / R_3$':
            print("Found param")
            ax.set_xticks([0.15, 0.25])
        if ax.get_ylabel() == '$R_2 / R_3$':
            ax.set_yticks([0.1, 0.2])

for ax in g.subplots.flatten():
    if ax is not None:
        if ax.get_xlabel() == '$\epsilon_\mathrm{AGN}$':
            print("Found param")
            ax.set_xticks([0.2, 0.4])
        if ax.get_ylabel() == '$\epsilon_\mathrm{AGN}$':
            ax.set_yticks([0.2, 0.4]) 
'''


# Define the position and size of the new subplot block (top-right corner)
left = 0.58
bottom = 0.62
width = 0.4
height = 0.36

ax = fig.add_axes([left, bottom, width, height])

sizes.plotMstarRstar(Ms = np.geomspace(3e8,1e12,100), xlim=[3e8,1e12], ylim=[0.07,300], ax=ax, write_chi2=False)
ax.set_ylim(0.5,200)
ax.set_yticks([1,10,100])
ax.set_yticklabels([1,10,100])
ax.set_xlim(5e8, 6e11)
ax.legend(loc='upper right', frameon=False)

g.export('paperPlotFigures/cornerPlotGetDist_fullCalibration.pdf')

#### Now, make separate SHMR plot
#####################################
set_latex_style()

fig = plt.figure(figsize=(6, 12))
gs = gridspec.GridSpec(5, 1, height_ratios=[1, 1, 0.4, 1, 1])

# Top group (z=0.3)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax1.set_title('z=0.29', size=22)

# Bottom group (z=0.9)
ax3 = fig.add_subplot(gs[3])
ax4 = fig.add_subplot(gs[4])
ax3.set_title('z=0.88', size=22)



# Plot the low-z SHMR on the axes
SHMRdata = SHMR.SHMR_lowz; xshift=0.01; SHMRscatterData = SHMR.SHMRscatter_lowz
inc = SHMRdata['massStellarLog10Target'] != 0
ax1.errorbar(np.log10(SHMRdata['massHalo'])[inc], SHMRdata['massStellarLog10Target'][inc]-np.log10(SHMRdata['massHalo'][inc]),
                        np.sqrt(np.diag(SHMRdata['massStellarLog10CovarianceTarget']))[inc],
                        label='L12 (target)', ls='', marker='o', color='k', lw=2)
ax1.errorbar(np.log10(SHMRdata['massHalo'])[inc]+xshift, SHMRdata['massStellarLog10'][inc]-np.log10(SHMRdata['massHalo'][inc]),
                    np.sqrt(np.diag(SHMRdata['massStellarLog10Covariance']))[inc],
                    label=r'max-$\mathcal{L}$', ls='', marker='o', color='C0', lw=2)
ax1.legend(loc='upper left', frameon=False)

ax2.errorbar(np.log10(SHMRscatterData['massHalo'])[inc], SHMRscatterData['massStellarLog10ScatterTarget'][inc],
            np.sqrt(np.diag(SHMRscatterData['massStellarLog10ScatterCovarianceTarget']))[inc], ls='', marker='o', color='k', lw=2)
ax2.errorbar(np.log10(SHMRscatterData['massHalo'])[inc]+xshift, SHMRscatterData['massStellarLog10Scatter'][inc],
            np.sqrt(np.diag(SHMRscatterData['massStellarLog10ScatterCovariance']))[inc], ls='', marker='o', color='C0', lw=2)


# Plot the high-z SHMR on the axes
SHMRdata = SHMR.SHMR_highz; xshift=0.01; SHMRscatterData = SHMR.SHMRscatter_highz
inc = SHMRdata['massStellarLog10Target'] != 0
ax3.errorbar(np.log10(SHMRdata['massHalo'])[inc], SHMRdata['massStellarLog10Target'][inc]-np.log10(SHMRdata['massHalo'][inc]),
                        np.sqrt(np.diag(SHMRdata['massStellarLog10CovarianceTarget']))[inc],
                        label='L12 (target)', ls='', marker='o', color='k', lw=2)
ax3.errorbar(np.log10(SHMRdata['massHalo'])[inc]+xshift, SHMRdata['massStellarLog10'][inc]-np.log10(SHMRdata['massHalo'][inc]),
                    np.sqrt(np.diag(SHMRdata['massStellarLog10Covariance']))[inc],
                    label=r'max-$\mathcal{L}$', ls='', marker='o', color='C0', lw=2)
#ax3.legend(loc='upper left', frameon=False)
ax4.errorbar(np.log10(SHMRscatterData['massHalo'])[inc], SHMRscatterData['massStellarLog10ScatterTarget'][inc],
            np.sqrt(np.diag(SHMRscatterData['massStellarLog10ScatterCovarianceTarget']))[inc], ls='', marker='o', color='k', lw=2)
ax4.errorbar(np.log10(SHMRscatterData['massHalo'])[inc]+xshift, SHMRscatterData['massStellarLog10Scatter'][inc],
            np.sqrt(np.diag(SHMRscatterData['massStellarLog10ScatterCovariance']))[inc], ls='', marker='o', color='C0', lw=2)




for ax in [ax1,ax2,ax3,ax4]:
    ax.set_xlim(11.41,12.59)

for ax in [ax1,ax3]:
    ax.set_xticks([])
    ax.set_ylim(-1.99, -1.31)
    ax.set_xlabel(None)
    ax.set_ylabel(r'$\left< \log_{10} M_\star / M_{200} \right>$')


for ax in [ax2,ax4]:
    ax.set_xticks([11.6,11.8,12.0,12.2,12.4])
    ax.set_xlabel(r'$\log_{10} M_{200\mathrm{m}} / \mathrm{M}_\odot$')
    ax.set_ylabel(r'$\sigma_{\log_{10} M_\star}$')
    ax.set_ylim(0,0.29)

fig.subplots_adjust(top=0.96, bottom=0.06, left=0.16, right=0.99, hspace=0.05, wspace=0.05)
fig.savefig('paperPlotFigures/fullCalibration_SHMR_singleLikelihood.pdf')