import h5py
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import corner
from matplotlib.colors import Normalize
import os
import xml.etree.ElementTree as ET
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import emcee

def h5Obj_to_dict(h5_obj):
    result = {}
    # Include attributes
    if hasattr(h5_obj, "attrs"):
        result["attributes"] = {key: h5_obj.attrs[key] for key in h5_obj.attrs}
    if isinstance(h5_obj, h5py.Group):
        for key, item in h5_obj.items():
            result[key] = h5Obj_to_dict(item)
    elif isinstance(h5_obj, h5py.Dataset):
        result = h5_obj[()]
    else:
        raise TypeError(f"Unsupported HDF5 type: {type(h5_obj)}")
    return result

def corner_scatter_likelihood(data, logLikelihoods, logLrange=4, scatter_kwargs={}, **kwargs):
    log_Lmax = np.max(logLikelihoods)
    vmin = log_Lmax - logLrange; vmax = log_Lmax
    # Normalize the log-likelihoods to the defined range
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    # make the plot
    fig = corner.corner(data, **kwargs); axes = np.array(fig.axes)
    # Overlay scatter plots on the off-diagonal plots
    ndim = data.shape[1]
    for i in range(ndim):
        for j in range(i):
            ax = axes[i * ndim + j]  # Access subplot (i, j)
            scatter = ax.scatter(data[:, j], data[:, i], c=logLikelihoods, norm=norm, **scatter_kwargs)        
    # Add a new axis for the colorbar
    cbar_ax = fig.add_axes([0.92, 0.35, 0.03, 0.55])  # [left, bottom, width, height] in figure coordinates
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label(r'$\log (\mathcal{L} / \mathcal{L}_\mathrm{max})$')
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([str(-logLrange), '0'])
    return fig
    
MCMCparams = ['velocityCharacteristic', 'exponent',
             'diskVelocityCharacteristic', 'diskExponent',
             'spheroidVelocityCharacteristic', 'spheroidExponent',
             'fracAngMomSpheroid', 'fracLossAngMom', 'fracAngMomRetSpheroid',
             'gamma', 'delta1', 'delta2',
             'starFormationFrequencyNormalization',
             'efficiencyRadioMode',
        ]

nonParameterColumnNames = ["stepNumber", "chainNumber", "tCPU", "converged", "logPosterior", "logLikelihood"]

MCMCparameterDict = {
    "nodeOperator/nodeOperator[@value='stellarFeedbackDisks']/stellarFeedbackOutflows/stellarFeedbackOutflows/velocityCharacteristic": "diskVelocityCharacteristic",
    "nodeOperator/nodeOperator[@value='stellarFeedbackDisks']/stellarFeedbackOutflows/stellarFeedbackOutflows/exponent": "diskExponent",
    "nodeOperator/nodeOperator[@value='stellarFeedbackSpheroids']/stellarFeedbackOutflows/stellarFeedbackOutflows/velocityCharacteristic": "spheroidVelocityCharacteristic",
    "nodeOperator/nodeOperator[@value='stellarFeedbackSpheroids']/stellarFeedbackOutflows/stellarFeedbackOutflows/exponent": "spheroidExponent",
    "blackHoleWind/efficiencyWind": "BHefficiencyWind",
    "blackHoleCGMHeating/efficiencyRadioMode": "BHefficiencyRadioMode",
    "hotHaloOutflowReincorporation/gamma": "henriquesGamma",
    "hotHaloOutflowReincorporation/delta1": "henriquesDelta1",
    "hotHaloOutflowReincorporation/delta2": "henriquesDelta2",
    "componentDisk/ratioAngularMomentumSolverRadius": "diskRatioAngularMomentumSolverRadius",
    "componentSpheroid/ratioAngularMomentumScaleRadius": "spheroidRatioAngularMomentumScaleRadius",
    "starFormationRateSurfaceDensityDisks/starFormationFrequencyNormalization": "normalisationBlitzSFR",
    "componentHotHalo/fractionLossAngularMomentum": "recycledFracLossAngMom",
    "nodeOperator/nodeOperator[@value='barInstability']/galacticDynamicsBarInstability/fractionAngularMomentumRetainedSpheroid": "barInstabilityFracAngMomRetSpheroid",
    "coolingRate/multiplier": "coolingRateMultiplier",
}

class MCMC_results:

    def __init__(self, outpath, paramNames='auto', runAll=False, burnLen=0.1, logName="mcmcChains_????.log", fname='mcmcConfig.xml'):
        self.outpath = outpath
        if paramNames == 'auto':
            self.paramNames = [MCMCparameterDict[paramName] for paramName in self.getParameterPaths(fname)]
        else:
            self.paramNames = paramNames
        self.column_names = nonParameterColumnNames + self.paramNames
        self.file_paths = sorted(glob.glob(self.outpath+"/"+logName))
        self.Nchain = len(self.file_paths)
        self.Ncol = self.getNcol(self.file_paths[0])
        if self.Ncol != len(self.column_names):
            raise ValueError(f"Number of columns does not match the number of column labels: {self.Ncol} != {len(self.column_names)}")
        self.burnLen = burnLen
        if runAll:
            self.readFiles()
            _ = self.cornerPlot()
            self.tracePlot()

    def getParameterPaths(self, fname='mcmcConfig.xml'):
        input_filename = os.path.join(self.outpath, fname)
        tree = ET.parse(input_filename)
        root = tree.getroot()
        # Find all "modelParameter" elements that are active
        parameters = []
        for param in root.findall(".//modelParameter[@value='active']"):
            name_element = param.find("name")
            if name_element is not None:
                parameters.append(name_element.attrib["value"])
        return parameters


    def getNcol(self, fname):
        # Determine the number of columns by reading the first line
        with open(fname, 'r') as f:
            first_line = f.readline()
            n_columns = len(first_line.split())  # Count the number of whitespace-separated values
            return n_columns

    def readFiles(self, burnLen=None, on_bad_lines='skip'):
        if burnLen is not None:
            self.burnLen = burnLen
        # Read and concatenate all files
        self.dataframes = [pd.read_csv(file, header=None, sep=r'\s+', names=self.column_names, on_bad_lines=on_bad_lines) for file in self.file_paths] # keep burn-in for plotting chain eveolution
        if self.burnLen < 1:
            # burnLen is the fraction of the chain to discard
            self.burnLen = int(self.burnLen * len(self.dataframes[0]))    
        self.combined_df = pd.concat([df.iloc[self.burnLen:] for df in self.dataframes], ignore_index=True) # remove burn-in for making corner plots, etc.
        
    def cornerPlot(self, scatterLikelihoods=False, logLrange=4, scatter_kwargs={}, colorByLogPosterior=False, **kwargs):
        samples = self.combined_df[self.paramNames]
        if scatterLikelihoods:
            colorBy = self.combined_df['logLikelihood']
            if colorByLogPosterior:
                colorBy = self.combined_df['logPosterior']
            return corner_scatter_likelihood(samples.to_numpy(), logLikelihoods=colorBy, logLrange=logLrange, scatter_kwargs=scatter_kwargs, labels=self.paramNames, **kwargs)
        else:
            return corner.corner(samples, labels=self.paramNames, **kwargs)
        
    def plot_corner(self, lib="getdist", parameters=None, truths=None, **kwargs):
        """
        Plot a corner plot using getdist or chainconsumer.

        Parameters
        ----------
        lib : str
            Library to use: "getdist" or "chainconsumer".
        parameters : list of str, optional
            Subset of parameter names to include. Defaults to all.
        burnin : int, optional
            Number of initial samples to discard.
        thin : int, optional
            Thinning factor.
        truths : list, optional
            List of true values to show on the plot.
        kwargs : dict
            Additional kwargs passed to the plotting library.
        """
        
        if parameters is None:
            parameters = self.paramNames
        idxs = [parameters.index(p) for p in parameters]
        samples = self.combined_df[parameters].to_numpy()
        flat_samples = samples.reshape(-1, samples.shape[-1])[:, idxs]

        if lib.lower() == "getdist":
            from getdist import MCSamples, plots

            gds = MCSamples(samples=flat_samples, names=parameters, labels=[f"{p}" for p in parameters])
            if truths:
                for name, val in zip(parameters, truths):
                    gds.addDerived(name=name, expr=None, label=f"${name}$", range=None, value=val)

            g = plots.get_subplot_plotter()
            g.triangle_plot(gds, filled=True, **kwargs)
            plt.show()
        
        elif lib.lower() == "chainconsumer":
            from chainconsumer import ChainConsumer

            c = ChainConsumer()
            c.add_chain(flat_samples, parameters=parameters, name="Posterior", truth=truths)
            c.configure(statistics="max", kde=True, **kwargs)
            fig = c.plot(figsize=(3.0 * len(parameters), 3.0 * len(parameters)))
            return fig

        else:
            raise ValueError(f"Unsupported library: {lib}. Choose 'getdist' or 'chainconsumer'.")

        
    def getAutocorrelationLength(self, params=None, printOutput=False):
        if params is None:
            params = self.paramNames
        params = np.atleast_1d(params)
        samples = self.combined_df[params].to_numpy()
        tau = emcee.autocorr.integrated_time(samples, quiet=True, has_walkers=False)
        if printOutput:
            for i, param in enumerate(params):
                print(f"Autocorrelation length for {param}: {tau[i]:.1f}")
        return tau


    def tracePlot(self, tightLayout=True): 
        fig, axs = plt.subplots(nrows=len(self.paramNames), figsize=(6, 0.5+len(self.paramNames)))
        axs = np.atleast_1d(axs)
        for j, param in enumerate(self.paramNames):
            ax = axs[j]
            for i in np.arange(self.Nchain):
                ax.plot(self.dataframes[i][param])
            ax.set_ylabel(param)
            if j == len(self.paramNames)-1: 
                ax.set_xlabel("step")
            else:
                ax.set_xticklabels([])
            ax.axvspan(0, self.burnLen, color='black', alpha=0.1)
            ax.set_xlim(xmin=0)
            ### Add autocorrelation length (tau) label
            tau_val = self.getAutocorrelationLength(params=param)[0]
            ax.text(0.99, 0.95, f"$\\tau$ = {tau_val:.1f}", transform=ax.transAxes,
                    ha='right', va='top', fontsize=10, color='gray',
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.2'))

        if tightLayout:
            fig.tight_layout()
        return fig
    
    def getMaxLikelihood(self,printOutput=False):
        if printOutput:
            print(self.combined_df.iloc[self.combined_df['logLikelihood'].argmax()].to_string(float_format='{:g}'.format))
        return self.combined_df.iloc[self.combined_df['logLikelihood'].argmax()]
    
    def getMaxPosterior(self,printOutput=False):
        if printOutput:
            print(self.combined_df.iloc[self.combined_df['logPosterior'].argmax()].to_string(float_format='{:g}'.format))
        return self.combined_df.iloc[self.combined_df['logPosterior'].argmax()]
    
    def getPosteriorMean(self,printOutput=False):
        if printOutput:
            print(self.combined_df.mean(numeric_only=True).to_string(float_format='{:g}'.format))
        return self.combined_df.mean(numeric_only=True)

    def saveGalacticusModelChangesConfigFile(self, parameterValueDict, fname='galacticusModelChanges.xml', output_dir='.', randomiseSeeds=False):
        # Create the new XML structure
        changes_root = ET.Element("changes")
        parameterPaths = self.getParameterPaths()
        parameterValues = [parameterValueDict[MCMCparameterDict[paramPath]] for paramPath in parameterPaths]
        for i in np.arange(len(parameterPaths)):
            ET.SubElement(changes_root, "change", type="update", path=parameterPaths[i], value=str(parameterValues[i]))
        if randomiseSeeds:
            # Add the randomNumberGenerator append block
            rng_change = ET.SubElement(changes_root, "change", type="replaceOrAppend", path="randomNumberGenerator")
            rng_element = ET.SubElement(rng_change, "randomNumberGenerator", value="GSL")
            ET.SubElement(rng_element, "seed", value=str(np.random.randint(0, 2**24)))
        # Write to an output file
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, fname)
        tree = ET.ElementTree(changes_root)
        ET.indent(tree, space="  ")  # Pretty-printing for readability
        tree.write(output_filename, encoding="UTF-8", xml_declaration=True)
        print(f"Generated file: {output_filename}")

    def saveGalacticusMaximumLikelihoodConfigFile(self, fname='maximumLikelihoodModelChanges.xml', output_dir='.'):
        self.saveGalacticusModelChangesConfigFile(self.getMaxLikelihood().to_dict(), fname=fname, output_dir=output_dir)

    def saveGalacticusMaximumPosteriorConfigFile(self, fname='maximumPosteriorModelChanges.xml', output_dir='.'):
        self.saveGalacticusModelChangesConfigFile(self.getMaxPosterior().to_dict(), fname=fname, output_dir=output_dir) 

    def saveGalacticusPosteriorMeanConfigFile(self, fname='posteriorMeanModelChanges.xml', output_dir='.'):
        self.saveGalacticusModelChangesConfigFile(self.getPosteriorMean().to_dict(), fname=fname, output_dir=output_dir)

    def saveGalacticusPosteriorDrawsConfigFiles(self, Ndraws=10, fname='posteriorDrawsModelChanges_{}.xml', seed=42, output_dir='.', randomiseSeeds=False):
        # Generate Ndraws files with random draws from the posterior
        self.randomSamples = self.combined_df.sample(n=Ndraws, random_state=seed)
        for i in np.arange(Ndraws):
            parameterValueDict = self.randomSamples.iloc[i].to_dict()
            self.saveGalacticusModelChangesConfigFile(parameterValueDict, fname=fname.format(i), output_dir=output_dir, randomiseSeeds=randomiseSeeds)

    def cornerPlotWithAnnotations(self, **kwargs):
        fig =  self.cornerPlot(**kwargs)
        # Get the axes from the figure (they're arranged in a grid)
        axes = np.array(fig.axes).reshape(len(self.paramNames), len(self.paramNames))
        # Loop through the sampled points and plot them on the correct subplots
        for ind, (idx, row) in enumerate(self.randomSamples.iterrows()):
            for i in range(len(self.paramNames)):
                for j in range(i):
                    ax = axes[i, j]
                    x = row[self.paramNames[j]]
                    y = row[self.paramNames[i]]
                    # Annotate with a number inside a solid circle
                    ax.annotate(
                        str(ind),
                        xy=(x, y),
                        ha='center',
                        va='center',
                        fontsize=8,
                        color='white',
                        weight='bold',
                        bbox=dict(boxstyle='circle,pad=0.2', facecolor='red', edgecolor='none', alpha=0.6)
                    )
        # draw on other points of interest
        for i in range(len(self.paramNames)):
            for j in range(i):
                ax = axes[i, j]
                meanP = self.getPosteriorMean().to_dict()
                ax.scatter(meanP[self.paramNames[j]], meanP[self.paramNames[i]], marker='s', color='cyan',s=50, zorder=1e3)
                maxP = self.getMaxPosterior().to_dict()
                ax.scatter(maxP[self.paramNames[j]], maxP[self.paramNames[i]], marker='+', color='magenta',s=70, lw=2, zorder=1e4)
                maxL = self.getMaxLikelihood().to_dict()
                ax.scatter(maxL[self.paramNames[j]], maxL[self.paramNames[i]], marker='x', color='orange',s=50, lw=2, zorder=1e5)
        # Choose an empty corner axis (top-right corner)
        legend_ax = axes[0, -1]
        legend_ax.axis('off')  # Hide the axis
        # Define legend handles
        handles = [
            Line2D([], [], marker='s', color='cyan', linestyle='None', label='Posterior Mean', markersize=8),
            Line2D([], [], marker='+', color='magenta', linestyle='None', label='Max Posterior', markersize=10),
            Line2D([], [], marker='x', color='orange', linestyle='None', label='Max Likelihood', markersize=8),
        ]
        # Create a proxy for the red circle with "i"
        posterior_sample_proxy = Circle((0, 0), radius=5, facecolor='red', edgecolor='none', alpha=0.6)
        posterior_sample_proxy_label = "Posterior Samples"
        # Use a custom handler to draw it as a red dot — text can't go in legend easily, so just describe it
        handles.append(posterior_sample_proxy)
        labels = [h.get_label() for h in handles[:-1]] + [posterior_sample_proxy_label]
        # Add the legend
        legend_ax.legend(handles=handles, labels=labels, loc='upper right', fontsize=8, frameon=False)
        return fig
    


class PSO_results(MCMC_results):

    def __init__(self, outpath, paramNames='auto', runAll=False, burnLen=0.1, logName="particleSwarmLogFileRoot_????.log", fname='particleSwarmConfig.xml'):
        self.outpath = outpath
        if paramNames == 'auto':
            self.paramNames = [MCMCparameterDict[paramName] for paramName in self.getParameterPaths(fname)]
            self.velocityNames = [MCMCparameterDict[paramName]+'_vel' for paramName in self.getParameterPaths(fname)]
        else:
            self.paramNames = paramNames
            self.velocityNames = [name+'_vel' for name in paramNames]
        self.column_names = nonParameterColumnNames + self.paramNames + self.velocityNames
        self.file_paths = sorted(glob.glob(self.outpath+"/"+logName))
        self.Nchain = len(self.file_paths)
        self.Ncol = self.getNcol(self.file_paths[0])
        if self.Ncol != len(self.column_names):
            raise ValueError(f"Number of columns does not match the number of column labels: {self.Ncol} != {len(self.column_names)}")
        self.burnLen = burnLen
        if runAll:
            self.readFiles()
            _ = self.cornerPlot()
            self.tracePlot()