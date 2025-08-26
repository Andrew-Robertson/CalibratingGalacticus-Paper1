import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import pandas as pd
from latex_plot_style import set_latex_style
from analysis.inference import singleLikelihoods as sl
import analysis.data_comparison.galacticusAutoAnalysis as analysis

set_latex_style()


dirs = {
    'lowz': '../galacticusMCMCs/Low-z',
    'highz': '../galacticusMCMCs/High-z',
    'lowz_highz': '../galacticusMCMCs/Low-z_High-z',
    'all':  '../galacticusMCMCs/Low-z_High-z_Sizes'
}
labels = {
    'lowz': 'Low-z',
    'highz': 'High-z',
    'lowz_highz': 'Low-z + High-z',
    'all': 'Low-z + High-z + Sizes'
}

lowz_filename = "romanEPS_singleLikelihood_lowz.hdf5"
highz_filename = "romanEPS_singleLikelihood_highz.hdf5"

fig, axs = plt.subplots(1,2,figsize=(8,6), gridspec_kw={'wspace': 0})

shifts = 1.05**np.array([-1.8,-1,1,1.8])
colours = [
    "#E41A1C",  # Red
    "#377EB8",  # Blue
    "#4DAF4A",  # Green
    "#984EA3",  # Purple
    "black"     # Data
]
markersize=4
alpha = 0.5
linewidth = 5
for i, key in enumerate(dirs.keys()):
    print(f"Loading {key}...")
    try:
        SHMR = sl.SHMR_likelihood(os.path.join(dirs[key],'singleLikelihood'), lowzFname=lowz_filename, highzFname=highz_filename)
        SHMR.load_data()
    except FileNotFoundError:
        SHMR = sl.SHMR_likelihood(os.path.join(dirs[key],'singleLikelihood/atBothLowAndHighZ'), lowzFname=lowz_filename, highzFname=highz_filename)
        SHMR.load_data()
    shift = shifts[i]
    col = colours[i]
    axs[0].errorbar(np.log10(SHMR.SHMR_lowz['massHalo']*shift), SHMR.SHMR_lowz['massStellarLog10'], SHMR.SHMRscatter_lowz['massStellarLog10Scatter'], marker='o', ls='', markersize=markersize, c=col)
    axs[1].errorbar(np.log10(SHMR.SHMR_highz['massHalo']*shift), SHMR.SHMR_highz['massStellarLog10'], SHMR.SHMRscatter_highz['massStellarLog10Scatter'], marker='o', ls='', markersize=markersize, c=col)
    # low-z lines
    inc = SHMR.SHMR_lowz['massStellarLog10']>0
    axs[0].plot(np.log10(SHMR.SHMR_lowz['massHalo'][inc]), SHMR.SHMR_lowz['massStellarLog10'][inc], lw=linewidth, alpha=alpha, c=col)
    # low-z lines
    inc = SHMR.SHMR_highz['massStellarLog10']>0
    axs[1].plot(np.log10(SHMR.SHMR_highz['massHalo'][inc]), SHMR.SHMR_highz['massStellarLog10'][inc], lw=linewidth, alpha=alpha, c=col, label=labels[key])
axs[0].errorbar(np.log10(SHMR.SHMR_lowz['massHalo']), SHMR.SHMR_lowz['massStellarLog10Target'], SHMR.SHMRscatter_lowz['massStellarLog10ScatterTarget'], marker='o', ls='', c='k', markersize=5, lw=2, label='Leauthaud et al. 2012')
axs[1].errorbar(np.log10(SHMR.SHMR_highz['massHalo']), SHMR.SHMR_highz['massStellarLog10Target'], SHMR.SHMRscatter_highz['massStellarLog10ScatterTarget'], marker='o', ls='', c='k', markersize=5, lw=2)

for ax in axs:
    ax.set_ylim(9.3,11.6)
    ax.legend(loc='upper left', frameon=False, fontsize=14)

axs[0].set_xlim(11.35, 12.65)
axs[1].set_xlim(11.45, 12.5)

axs[1].set_yticklabels([])
axs[0].set_ylabel(r'$\log_{10} M_\star / M_\odot$', labelpad=10, fontsize=18)
fig.supxlabel(r'$\log_{10} M_{\rm 200m} / M_\odot$', fontsize=18)

axs[0].set_title("z=0.29", fontsize=18)
axs[1].set_title("z=0.88", fontsize=18)

plt.tight_layout()

fig.savefig('paperPlotFigures/matchingMultipleDatasets.pdf')