import matplotlib.pyplot as plt
import numpy as np
import os
import analysis.data_comparison.galacticusAutoAnalysis as analysis
import h5py
import pandas as pd
from latex_plot_style import set_latex_style

set_latex_style()

titles = {
    0: 'z = 0.2 - 0.5',
    1: 'z = 0.5 - 0.75',
    2: 'z = 0.75 - 1',
    3: 'z = 1 - 1.25',
    4: 'z = 1.25 - 1.5',
    5: 'z = 1.5 - 2',
    6: 'z = 2 - 2.5'
}


def plotTomczak(bin=0, xlim=None, ylim=None, ax=None, color=None, shift_y_fac=1, title=True, legend=True):
    with h5py.File("../galacticusRuns/Low-z_High-z_Sizes/maximumPosterior/romanEPS_massFunction.hdf5") as f:
        pd = analysis.plot_data(f['analyses/massFunctionStellarTomczak2014ZFOURGEz'+str(bin)])
    Nsamp = 10
    percentileRange = 0.95
    posteriorSamplesData = np.zeros((Nsamp, len(pd.xData)))
    for i in np.arange(Nsamp):
        fname = f"../galacticusRuns/Low-z_High-z_Sizes/posteriorDraws/massFunction_{i}/romanEPS_massFunction.hdf5"
        with h5py.File(fname) as ff:
            mf = analysis.plot_data(ff['analyses/massFunctionStellarTomczak2014ZFOURGEz'+str(bin)])
            posteriorSamplesData[i] = mf.yData
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
    else:
        fig = ax.figure
    shift_y_fac *= np.log(10) # make per dex not per e-fold
    ax.errorbar(pd.xData, pd.yTarget*shift_y_fac, pd.yTargetErr*shift_y_fac, ls='', marker='o', c=color)
    ax.fill_between(pd.xData, np.percentile(posteriorSamplesData,50*(1-percentileRange),axis=0)*shift_y_fac, np.percentile(posteriorSamplesData,50*(1+percentileRange),axis=0)*shift_y_fac, color=color, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # now plot the Galacticus data
    ax.plot(pd.xData, pd.yData*shift_y_fac, color=color)
    ax.set_xlabel(r"$M_\star \, / \, \mathrm{M_\odot}$")
    ax.set_ylabel(r"$\Phi \, / \, \mathrm{Mpc}^{-3} \, \mathrm{dex}^{-1}$")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if legend:
        plt.legend(frameon=False, fontsize=12, loc='lower left')
    if title:
        plt.title(titles[bin], size=16)
    plt.tight_layout()
    return ax


cmap = plt.get_cmap('viridis')
# Get 4 evenly spaced colors from the colormap
colors = [cmap(i) for i in np.linspace(0.05, 0.95, 4)]

fig, ax = plt.subplots(figsize=(6,6))

fac = 1/4.
plotTomczak(bin=0, ax=ax, color=colors[0], title=False, legend=False)
plotTomczak(bin=2, ax=ax, color=colors[1], shift_y_fac=fac, title=False, legend=False)
plotTomczak(bin=4, ax=ax, color=colors[2], shift_y_fac=fac**2, title=False, legend=False)
plotTomczak(bin=6, ax=ax, color=colors[3], shift_y_fac=fac**3, title=False, legend=False)

ax.set_xlim(8e7, 4e11)
ax.set_ylim(2e-7, 2e-1)

# Dummy data for colored lines
z_labels = [r"$0.2 < z < 0.5$", r"$0.75 < z < 1 \,\, (\Phi \times 1/4)$", r"$1.25 < z < 1.5 \,\, (\Phi \times 1/16)$", r"$2 < z < 2.5 \,\, (\Phi \times 1/64)$"]
z_handles = []
for i, (label, color) in enumerate(zip(z_labels, colors)):
    line, = ax.plot([0, 1], [i, i], color=color, label=label)
    z_handles.append(line)

# Galacticus line and data errorbar
galacticus_line, = ax.plot([0, 1], [4, 4], color='black')
data_err = ax.errorbar(0.5, 3.5, yerr=0.2, fmt='o', color='black')
galacticus_default_line, = ax.plot([0, 1], [4, 4], color='black', ls='--')

# First legend (top-right)
legend1 = ax.legend(
    handles=[galacticus_line, data_err],
    labels=['Galacticus', 'ZFOURGE/CANDELS'],
    loc='upper right', frameon=False, fontsize=14
)
ax.add_artist(legend1)

# Bottom-left legend (redshift bins)
legend2 = ax.legend(handles=z_handles, loc='lower left', title="Redshift bins", frameon=False, fontsize=14, title_fontsize=14)


fig.savefig('paperPlotFigures/ZFOURGE_stellarMassFuncComparison.pdf')