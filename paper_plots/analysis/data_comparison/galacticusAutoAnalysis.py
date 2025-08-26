import h5py
import matplotlib.pyplot as plt
import numpy as np

def hdf5_attrs_to_dict(hdf5_obj):
    # Convert HDF5 attributes to a dictionary
    return {key: value for key, value in hdf5_obj.attrs.items()}

def hdf5_group_to_dict(hdf5_group):
    # Convert the datasets in the HDF5 group to a dictionary
    group_dict = {}
    for key, dataset in hdf5_group.items():
        if isinstance(dataset, h5py.Dataset):  # Ensure it's a dataset
            group_dict[key] = dataset[:]  # Read the dataset into a NumPy array
    return group_dict

class plot_data():

    def __init__(self, g):
        # g is an hdf5 group
        attrsDict = hdf5_attrs_to_dict(g)
        dataDict = hdf5_group_to_dict(g)
        if attrsDict['type'].decode('utf-8') != 'function1D':
            raise NotImplementedError()
        # get the data
        self.xData = dataDict[attrsDict['xDataset'].decode('utf-8')]
        self.yData = dataDict[attrsDict['yDataset'].decode('utf-8')]
        self.yDataErr = np.sqrt(np.diag(dataDict[attrsDict['yCovariance'].decode('utf-8')]))
        try:
            self.yTarget = dataDict[attrsDict['yDatasetTarget'].decode('utf-8')]
            try:
                self.yTargetErr = np.sqrt(np.diag(dataDict[attrsDict['yCovarianceTarget'].decode('utf-8')]))    
            except KeyError:
                self.yTargetErr = np.zeros(len(self.yTarget))
            self.targetLabel = attrsDict['targetLabel'].decode('utf-8')   
        except KeyError:
            self.yTarget = None
            self.yTargetErr = None
            self.targetLabel = None

        self.xAxisLabel = attrsDict['xAxisLabel'].decode('utf-8')
        self.yAxisLabel = attrsDict['yAxisLabel'].decode('utf-8')
        self.xAxisIsLog = attrsDict['xAxisIsLog']
        self.yAxisIsLog = attrsDict['yAxisIsLog']


def plot_from_analysis_group(g, ax=None, savefig=None, buffer=0.2, plotTarget=True, **kwargs):
    # g is an hdf5 group
    # savefig should be the name of a file in which to save the figure (if you want to save it)
    # buffer is a fractional amount of additional space at the top and bottom of the figure
    pd = plot_data(g)
    targetDataExists=True
    if pd.yTarget is None:
        targetDataExists = False
    # now plot it
    if ax is None:
        fig, ax = plt.subplots()
    if targetDataExists:
        # plot it, and use it to inform axis limits
        if plotTarget:
            ax.errorbar(pd.xData, pd.yTarget, pd.yTargetErr, ls='', marker='o', label='target')
        if pd.xAxisIsLog:
            ax.set_xscale('log')
        if pd.yAxisIsLog:
            ax.set_yscale('log')
        # Capture the limits based on the target data
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()  
        # Check if the y-axis is set to log scale
        if ax.get_yscale() == 'log':
            # In log scale, we multiply the limits to add a buffer
            buffer_factor = 10**(buffer*np.log10(ylims[1]/ylims[0]))
            new_ylims = (ylims[0] / buffer_factor, ylims[1] * buffer_factor)
        else:
            # In linear scale, we add/subtract a fixed buffer
            y_range = ylims[1] - ylims[0]
            buffer_size = buffer * y_range  # 20% of the y-range
            new_ylims = (ylims[0] - buffer_size, ylims[1] + buffer_size)
    else:
        if pd.xAxisIsLog:
            ax.set_xscale('log')
        if pd.yAxisIsLog:
            ax.set_yscale('log')
    # now plot the Galacticus data
    default_kwargs = {'ls': '', 'marker': 'o', 'label': 'Galacticus'}
    # Update with any overrides from kwargs
    default_kwargs.update(kwargs)
    ax.errorbar(pd.xData, pd.yData, pd.yDataErr, **default_kwargs)
    ax.set_xlabel(pd.xAxisLabel)
    ax.set_ylabel(pd.yAxisLabel)
    if targetDataExists:
        ax.set_xlim(xlims)
        ax.set_ylim(new_ylims)
    plt.title(pd.targetLabel)  
    if savefig is not None:
        plt.savefig(savefig)  

def plot_all_galacticus_analyses(fname, savefigures=False, savedir='.'):
    # fname is a galacticus *.hdf5 file
    with h5py.File(fname) as f:
        analysis_names = list(f['analyses'].keys())
        for analysis_name in analysis_names:
            savefig=None
            if savefigures:
                savefig = savedir+'/'+analysis_name+".png"
            plot_from_analysis_group(f['analyses'][analysis_name], ax=None, savefig=savefig)


