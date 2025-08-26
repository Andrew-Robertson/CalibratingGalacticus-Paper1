import matplotlib.pyplot as plt
import numpy as np
import analysis.data_comparison.galacticusAutoAnalysis as analysis
import h5py
import pandas as pd
from latex_plot_style import set_latex_style

set_latex_style()

phi = []
count_in_bin = []

fname = "../galacticusRuns/Low-z_High-z_Sizes/maximumPosterior/sobralComparison/romanEPS_massFunction.hdf5"

with h5py.File(fname) as f:
    # Output4 is z=0.4
    for OutputGr in [4,3,2,1]:
        nd = analysis.hdf5_group_to_dict(f["/Outputs/Output"+str(OutputGr)+"/nodeData"])
        weights = f["/Outputs/Output"+str(OutputGr)+"/mergerTreeWeight"][:]
        HalphaL = nd["luminosityEmissionLine:balmerAlpha6565Panuzzo"]
        # calc luminosity functions
        volume = 1/np.mean(weights)
        counts, bin_edges = np.histogram(np.log10(HalphaL), bins=20, range=[40,44])
        bin_mids = 0.5*(bin_edges[1:] + bin_edges[:-1])
        bin_width = (bin_edges[1] - bin_edges[0])
        phi.append(counts / (volume * bin_width))
        count_in_bin.append(counts)

with h5py.File(fname) as f:
    data = [analysis.plot_data(f['/analyses/luminosityFunctionHalphaSobral2013HiZELSZ'+str(ind)]) for ind in np.arange(4)+1]
for pd in data:
    pd.yAxisLabel = r'$\mathrm{d}n / \mathrm{d} \log_{10} L_{\mathrm{H} \alpha} \,\, / \,\, \mathrm{Mpc}^{-3}$'
    pd.xAxisLabel = r'$L_{\mathrm{H} \alpha} \,\, / \,\, \mathrm{erg \, s}^{-1}$'
    pd.yTarget *= np.log(10); pd.yTargetErr *= np.log(10); pd.yData *= np.log(10); pd.yDataErr *= np.log(10)

fig, ax = plt.subplots(figsize=(6,5))
cols = ['gray', 'green', 'red', 'blue']
markers = ['s', 'd', '*', 'o']
dataLabels = ['z=0.4 (HiZELS)', 'z=0.84 (HiZELS)', 'z=1.47 (HiZELS)', 'z=2.23 (HiZELS)']
shifts = np.array([-3,-1,1,3])*0.01
for i, pd in enumerate(data):
    ax.errorbar(pd.xData*(10**shifts[i]), pd.yTarget, pd.yTargetErr, ls='', marker=markers[i], label=dataLabels[i], color=cols[i], markersize=6, markeredgecolor='black')
    #ax.plot(pd.xData*(10**shifts[i]), pd.yData, color=cols[i], lw=3)
    N = count_in_bin[i]
    ax.plot(10**bin_mids*(10**shifts[i]), phi[i], color=cols[i], lw=3, ls='-', alpha=0.3)
    ax.plot((10**bin_mids*(10**shifts[i]))[N>10], phi[i][N>10], color=cols[i], lw=3, ls='-')

ax.set_xlabel(pd.xAxisLabel)
ax.set_ylabel(pd.yAxisLabel)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(10**40.4, 5e43)
ax.set_ylim(10**-5.2, 5e-2)
leg1 = ax.legend(frameon=False, fontsize=14)
gal_line, = plt.plot([1e22,1e23],[5,5], c='k', lw=3, label='Galactcius')
ax.legend([gal_line], ['Galacticus'], loc='upper right', frameon=False, fontsize=14)
ax.add_artist(leg1)  # Add it manually
plt.tight_layout()

plt.savefig('paperPlotFigures/Sobral13_hAlphaComparison.pdf')