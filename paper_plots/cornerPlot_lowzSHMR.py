import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import pandas as pd
from latex_plot_style import set_latex_style
from getdist import MCSamples, plots
from analysis.inference import singleLikelihoods as sl
import analysis.data_comparison.galacticusAutoAnalysis as analysis
from analysis.inference import mcmc

set_latex_style()

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

basePath = "../galacticusMCMCs/Low-z" 

results = mcmc.MCMC_results(os.path.join(basePath,"run"), burnLen=0.1)
results.readFiles()
SHMR = sl.SHMR_likelihood(os.path.join(basePath,"singleLikelihood"), lowzFname="romanEPS_singleLikelihood_lowz.hdf5", highzFname=None)
SHMR.load_data()

parameters = results.paramNames
idxs = [parameters.index(p) for p in parameters]
samples = results.combined_df[parameters].to_numpy()[:, idxs]

ranges={'diskExponent':(0.0, 5.0), 'henriquesGamma':(1.0, 25.0), 'henriquesDelta1':(-2.0,4.0), 'henriquesDelta2':(-1.0,5.0), 'BHefficiencyRadioMode':(0.03,1.0), 'spheroidRatioAngularMomentumScaleRadius':(0.1,0.5)}
labels = [r'V_\mathrm{disk} \, / \, \mathrm{km \, s^{-1}}', r'\alpha_\mathrm{disk}', r'\gamma', r'\delta_1', r'\delta_2', r'\epsilon_\mathrm{AGN}', r'R_2 / R_3']



gds_smooth = MCSamples(samples=samples, names=parameters, labels=labels, settings={"smooth_scale_1D": 5.0, "smooth_scale_2D": 3.0}, ranges=ranges)
gds = MCSamples(samples=samples, names=parameters, labels=[f"{p}" for p in parameters])

color = 'gray'
g = plots.get_subplot_plotter()
g.settings.axes_fontsize = 30
g.settings.axes_labelsize = 30
#g.triangle_plot([gds, gds_smooth], filled=True)
g.triangle_plot([gds_smooth], filled=True, colors=[color,])
# Manually update the 1D line color
for ax in g.subplots.flatten():
    if ax is not None:
        for line in ax.lines:
            line.set_color(color)

fig = g.fig  # Get the figure object

# Define the position and size of the new subplot block (top-right corner)
left = 0.58
bottom = 0.62
width = 0.4
height = 0.36

# Height ratio between the two plots
gap = 0.005
h1 = (height - gap) * 0.5
h2 = (height - gap) * 0.5

# Bottom axis (shared x)
ax_bottom = fig.add_axes([left, bottom, width, h1])
# Top axis (no x-axis ticks)
ax_top = fig.add_axes([left, bottom + h1 + gap, width, h2])

# Plot the SHMR on the axes
SHMRdata = SHMR.SHMR_lowz; xshift=0.01; SHMRscatterData = SHMR.SHMRscatter_lowz
inc = SHMRdata['massStellarLog10Target'] != 0
ax_top.errorbar(np.log10(SHMRdata['massHalo'])[inc], SHMRdata['massStellarLog10Target'][inc]-np.log10(SHMRdata['massHalo'][inc]),
                        np.sqrt(np.diag(SHMRdata['massStellarLog10CovarianceTarget']))[inc],
                        label='L12 (target)', ls='', marker='o', color='k', lw=2)
ax_top.errorbar(np.log10(SHMRdata['massHalo'])[inc]+xshift, SHMRdata['massStellarLog10'][inc]-np.log10(SHMRdata['massHalo'][inc]),
                    np.sqrt(np.diag(SHMRdata['massStellarLog10Covariance']))[inc],
                    label=r'max-$\mathcal{L}$', ls='', marker='o', color='C0', lw=2)
ax_top.legend(loc='upper left', frameon=False)
ax_top.set_xlabel(None)
ax_top.set_xticks([])

ax_bottom.errorbar(np.log10(SHMRscatterData['massHalo'])[inc], SHMRscatterData['massStellarLog10ScatterTarget'][inc],
            np.sqrt(np.diag(SHMRscatterData['massStellarLog10ScatterCovarianceTarget']))[inc], ls='', marker='o', color='k', lw=2)
ax_bottom.errorbar(np.log10(SHMRscatterData['massHalo'])[inc]+xshift, SHMRscatterData['massStellarLog10Scatter'][inc],
            np.sqrt(np.diag(SHMRscatterData['massStellarLog10ScatterCovariance']))[inc], ls='', marker='o', color='C0', lw=2)


ax_bottom.set_xlabel(r'$\log_{10} M_{200\mathrm{m}} / \mathrm{M}_\odot$')
ax_bottom.set_ylabel(r'$\sigma_{\log_{10} M_\star}$')
ax_top.set_ylabel(r'$\left< \log_{10} M_\star / M_{200} \right>$')

ax_bottom.set_ylim(0,0.29)
ax_top.set_ylim(-1.83, -1.31)

for ax in [ax_top, ax_bottom]:
    ax.set_xlim(11.41,12.59)

ax_bottom.set_xticks([11.6,11.8,12.0,12.2,12.4])
# ax_top.set_yticks([11.6,11.8,12.0,12.2,12.4])

g.export('paperPlotFigures/cornerPlotGetDist_lowzSHMR.pdf')