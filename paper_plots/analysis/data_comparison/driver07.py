import h5py
import dust_model.external_data as data
from dust_model import observing
from dust_model import galacticus
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Driver study primarily done in the B-band, let's get an effective wavelength of the B-band
wav, trans = observing.read_galacticus_filter('SuprimeCam_B') # wav in angstroms
Bwav = np.average(wav, weights=trans)

class diffuse_dust_attenuation_interpolator():
     
    def __init__(self, AttenuationModel='Benson2018', interpolator_bounds_error=True):
          self.AttenuationFname = data.diffuse_dust_attenuation_files[AttenuationModel]
          self.interpolator_bounds_error = interpolator_bounds_error
          self.build_attenuation_interpolators()

    def build_attenuation_interpolators(self,):

        with h5py.File(self.AttenuationFname) as f:
            self.Attenuation_wavelength = f['wavelength'][:]
            self.Attenuation_inclination = f['inclination'][:]
            self.Attenuation_opticalDepth = f['opticalDepth'][:]
            self.Attenuation_spheroidScaleRadial = f['spheroidScaleRadial'][:]
            self.Attenuation_attenuationDisk = f['attenuationDisk'][:]
            self.Attenuation_attenuationSpheroid = f['attenuationSpheroid'][:]  
            
        # Make interpolator for disk attenuation
        disk_points = (self.Attenuation_wavelength, self.Attenuation_inclination, self.Attenuation_opticalDepth)
        self.diskInterpolatorParameters = ["wavelength / microns", "inclination / degrees", "optical depth (tauV0)"]
        disk_values = self.Attenuation_attenuationDisk
        self.attenuationDisk_interpolator = RegularGridInterpolator(disk_points, disk_values, bounds_error=self.interpolator_bounds_error, fill_value=1)

        # Make interpolator for spheroid attenuation
        spheroid_points = (self.Attenuation_wavelength, self.Attenuation_inclination, self.Attenuation_opticalDepth, self.Attenuation_spheroidScaleRadial)
        self.spheroidInterpolatorParameters = ["wavelength / microns", "inclination / degrees", "optical depth (tauV0)", "r_spheroid / r_disk"]
        spheroid_values = self.Attenuation_attenuationSpheroid
        self.attenuationSpheroid_interpolator = RegularGridInterpolator(spheroid_points, spheroid_values, bounds_error=self.interpolator_bounds_error, fill_value=1)

def get_binned_mean(x,y,xlim=None,bins=10):
    if xlim is None:
        xmin = x.min(); xmax = x.max()
    else:
        xmin = xlim[0]; xmax = xlim[1]
    bin_edges = np.linspace(xmin, xmax, num=bins+1) 
    # Digitize x values into bins
    bin_indices = np.digitize(x, bin_edges)
    # Calculate the mean and error on the mean (standard error) of y in each bin
    means_y = []
    errors_y = []
    for i in np.arange(1, len(bin_edges)):
        bin_values = y[bin_indices == i]
        mean_y = np.nanmean(bin_values)
        means_y.append(mean_y)
        # Calculate the standard error of the mean (SEM)
        n = np.sum(~np.isnan(bin_values))  # number of non-NaN values
        if n > 1:  # Avoid division by zero for bins with fewer than 2 points
            std_y = np.nanstd(bin_values)
            error_y = std_y / np.sqrt(n)
        else:
            error_y = np.nan  # If there's only one point or none, SEM is not defined
        errors_y.append(error_y)
    # Calculate the center of each bin
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, means_y, errors_y


def get_luminosity_function(mags, range=[-23,-17], bins=20, volume=1):
    # assume volume in Mpc^3, then phi in [counts / mag / Mpc^3]
    counts, bin_edges = np.histogram(mags, bins=bins, range=range)
    bin_mids = 0.5*(bin_edges[1:] + bin_edges[:-1])
    bin_width = bin_edges[1] - bin_edges[0]
    phi = counts / (volume * bin_width)
    phi_error = phi / np.sqrt(counts)
    phi_error[np.isnan(phi_error)] = np.inf
    return phi, bin_mids, phi_error

# Define the Schechter function in terms of magnitude M
def schechter_function(M, log10_phi_star, M_star, alpha):
    phi_star = 10**log10_phi_star
    term1 = 0.4 * np.log(10) * phi_star
    term2 = 10**(0.4 * (alpha + 1) * (M_star - M))
    term3 = np.exp(-10**(0.4 * (M_star - M)))
    return term1 * term2 * term3

# Function to fit the Schechter function to binned data
def fit_schechter(M, phi, phi_err, initial_guess = [3, -20.0, -1.0], fixed_alpha=None, print_best_fit=True):
    # Initial guess for the parameters: [log10_phi_star, M_star, alpha]

    # Fit the Schechter function to the data
    if fixed_alpha is not None:
        popt, pcov = curve_fit(lambda M, phi_star, M_star: schechter_function(M, phi_star, M_star, fixed_alpha), M, phi, p0=initial_guess[:-1], sigma=phi_err, absolute_sigma=True)
    else:
        popt, pcov = curve_fit(schechter_function, M, phi, p0=initial_guess, sigma=phi_err, absolute_sigma=True)
    
    # Print the best-fit parameters
    if print_best_fit:
        print("Best-fit parameters:")
        print(f"log10 phi_star = {popt[0]}")
        print(f"M_star = {popt[1]}")
        if fixed_alpha is None:
            print(f"alpha = {popt[2]}")
        else:
            print(f"alpha fixed to {fixed_alpha}")
    
    # Return the best-fit parameters and covariance matrix
    return popt, pcov

class luminosity_function():

    def __init__(self, mags, run=True, fixed_alpha=None, volume=None):
        # Volume in Mpc^3 (if provided)
        self.volume = volume
        self.mags = mags
        if run:
            self.calc_luminosity_function()
            self.fit_schechter(fixed_alpha=fixed_alpha)

    def calc_luminosity_function(self, range=[-23,-17], bins=20):
        self.range = range
        self.bins = bins
        if self.volume is None:
            volume = 1
        else:
            volume = self.volume
        self.phi, self.bin_mags, self.phi_error = get_luminosity_function(self.mags, range=range, bins=bins, volume=volume)

    def fit_schechter(self, initial_guess=[3, -20.0, -1.0], fixed_alpha=None, print=False):
        self.schechter_params, self.schechter_cov = fit_schechter(self.bin_mags, self.phi, self.phi_error, initial_guess = initial_guess, fixed_alpha=fixed_alpha, print_best_fit=print)
        if fixed_alpha is not None:
            # fill the params and cov with values related to alpha
            self.schechter_params = np.append(self.schechter_params, fixed_alpha)
            new_cov = np.zeros((3,3)); new_cov[:2,:2] = self.schechter_cov
            self.schechter_cov = new_cov
            
    def plot_results(self, ax=None, ylim=None, color='k', shift=0, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        mags = np.linspace(-23,-17,100)
        ax.errorbar(self.bin_mags+shift,self.phi,self.phi_error,ls='',marker='o', color=color, **kwargs)
        ax.plot(mags, schechter_function(mags, *self.schechter_params), color=color)
        ax.set_yscale('log')
        ax.set_ylim(ylim)
        ax.set_xlabel(r'$M_B$')
        if self.volume is None:
            ax.set_ylabel(r"$\phi \, / \, \mathrm{mergerTreeFile}^{-1} \, \mathrm{mag}^{-1}$")
        else:
            ax.set_ylabel(r"$\phi \, / \, \mathrm{Mpc}^{-3} \, \mathrm{mag}^{-1}$")
        return ax

def running_mean_bin(x, y, window_width, xs=None):
    """
    Calculate the running mean of y in bins of x, where the bins slide over the x values.

    Parameters:
    - x: array-like, the independent variable values (sorted in ascending order).
    - y: array-like, the dependent variable values.
    - window_width: float, the width of the sliding window in terms of x.

    Returns:
    - x_mean: array-like, the x values at the center of each window.
    - y_mean: array-like, the running mean of y values within each window.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    x_mean = []
    y_mean = []

    if xs is None:
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        print(xs)
    # Loop through each point in x and calculate the mean of y in a sliding window
    for i in range(len(xs)):
        # Define the window: from (x[i] - window_width/2) to (x[i] + window_width/2)
        lower_bound = xs[i] - window_width / 2
        upper_bound = xs[i] + window_width / 2
        
        # Select the indices within the window
        indices_in_window = np.where((x >= lower_bound) & (x <= upper_bound))[0]
        
        # Compute the mean x and mean y within this window if there are points inside the window
        if len(indices_in_window) > 0:
            x_mean.append(np.mean(x[indices_in_window]))
            y_mean.append(np.mean(y[indices_in_window]))

    return np.array(x_mean), np.array(y_mean)



class d07_comparison():
    ''' '''

    def __init__(self, galacticusFname, galacticusOutputGroups=None, AttenuationModel='Benson2018', interpolator_bounds_error=False, fcloud=0.25, seed=0, skip_init_functions=False):
        np.random.seed(seed)
        self.diffuseDust = diffuse_dust_attenuation_interpolator(AttenuationModel=AttenuationModel, interpolator_bounds_error=interpolator_bounds_error)
        self.galacticusFname = galacticusFname
        self.fcloud = fcloud
        self.load_data(output_groups=galacticusOutputGroups)
        if skip_init_functions:
            return
        self.calculate_attenuations()
        self.bulge_disk_decomposition()
        self.getMillenniumGalaxyCatalogue()
    
    def load_data(self,output_groups=None):
        fnames = glob.glob(self.galacticusFname)
        self.Roman = galacticus.galacticus_node_data(fnames, lightcone=False, output_groups=output_groups)
        selection = lambda g: galacticus.mstar_selection(g, Mmin=0) # selection of all galaxies with non-zero stellar mass
        self.Roman.load_data(selection=selection)

    def calculate_attenuations(self,):
        # inclinations between 0 and 90 degrees, uniformly from a sphere
        self.cos_inclinations = np.random.uniform(0,1,self.Roman.Ngal)
        self.inclinations = np.rad2deg(np.arccos(self.cos_inclinations)) 
        # calculate tauV0 for all galaxies
        self.Roman.calculate_optical_depth(fcloud = self.fcloud)
        self.tauV0 = self.Roman.data['tauV0']
        # calculate the attenuation of the spheroid and disk
        wav = Bwav * 1e-4 * np.ones(self.Roman.Ngal) # microns, as expected by interpolators
        self.diskAttenBband = self.diffuseDust.attenuationDisk_interpolator(np.stack((wav,self.inclinations,self.tauV0)).T)
        self.relativeSpheroidSize = self.Roman.data['spheroidRadius'] / self.Roman.data['diskRadius']
        self.spheroidAttenBband = self.diffuseDust.attenuationSpheroid_interpolator(np.stack((wav,self.inclinations,self.tauV0, self.relativeSpheroidSize)).T)     
        self.diskAttenBbandMag = -2.5*np.log10(self.diskAttenBband)
        self.spheroidAttenBbandMag = -2.5*np.log10(self.spheroidAttenBband)

    def bulge_disk_decomposition(self, BmagString='LuminositiesStellar:SuprimeCam_B:observed:z0.0000'):
        self.absoluteMagDisk = - 2.5*np.log10(self.Roman.data['disk'+BmagString])
        self.absoluteMagSpheroid = - 2.5*np.log10(self.Roman.data['spheroid'+BmagString])
        self.absoluteMagTotal = - 2.5*np.log10(self.Roman.data['spheroid'+BmagString] + self.Roman.data['disk'+BmagString])
        self.bulgeToTotal = self.Roman.data['spheroid'+BmagString] / (self.Roman.data['disk'+BmagString]+self.Roman.data['spheroid'+BmagString])

    def getMillenniumGalaxyCatalogue(self,):

        # Function to make parameter names unique if duplicates exist
        def make_unique(param_names):
            name_count = {}
            unique_names = []
            for name in param_names:
                if name in name_count:
                    name_count[name] += 1
                    unique_name = f"{name}_{name_count[name]}"  # Append suffix to duplicate
                else:
                    name_count[name] = 0
                    unique_name = name
                unique_names.append(unique_name)
            return unique_names

        # Load the parameter names and descriptions file
        def load_parameter_names(file_path):
            param_names = []
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.split(maxsplit=1)
                    if len(parts) > 0:
                        param_names.append(parts[0])  # Extract only the first part (the parameter name)
            # Make the parameter names unique
            param_names = make_unique(param_names)
            return param_names

        # Load the data file into a pandas DataFrame
        def load_data(file_path, param_names):
            df = pd.read_csv(file_path, header=None, names=param_names, sep=r'\s+')
            return df

        param_names_file = '/home/arobertson/Galacticus/Galacticus-dust-modelling/Galacticus-dust-modelling/datasets/morphology/Driver07/mgc_gim2d.par'  # Path to the parameter names file
        data_file = '/home/arobertson/Galacticus/Galacticus-dust-modelling/Galacticus-dust-modelling/datasets/morphology/Driver07/mgc_gim2d'  # Path to the data file
        # Load parameter names
        param_names = load_parameter_names(param_names_file)
        # Load the data into a pandas DataFrame
        alldata_df = load_data(data_file, param_names)
        # remove the "bad apples" and save the dataframe as an attribute of this object
        self.MGCdataframe = alldata_df[(alldata_df['BULGE_FRAC']<=1)*(alldata_df['ABS_MAG_GIM2DB']<0)]

    def plotBulgeToTotalComparison(self,ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(self.absoluteMagTotal, self.bulgeToTotal, s=2, alpha=0.5, label="Galacticus")
        ax.scatter(self.MGCdataframe['ABS_MAG_GIM2DB'], self.MGCdataframe['BULGE_FRAC'],s=2,alpha=0.5, label="MGC")
        ax.set_ylim(0,1)
        ax.set_xlim(-23,-16)
        ax.set_xlabel(r'$M_B$')
        ax.set_ylabel(r'$B/T$')
        ax.legend()
        return ax
    
    def plotDiskSizeComparison(self,ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(self.absoluteMagDisk, self.Roman.data['diskRadius']*1e3, s=2, alpha=0.5, label="Galacticus")
        ax.scatter(self.MGCdataframe['ABS_MAG_GIM2D'], self.MGCdataframe['R_D_KPC'],s=2,alpha=0.5, label="MGC")
        ax.set_ylim(1e-2, 1e2)
        ax.set_xlim(-23,-16)
        ax.set_xlabel(r'$M_B^\mathrm{disk}$')
        ax.set_ylabel(r'$R_\mathrm{disk} \, / \, \mathrm{kpc}$')
        ax.set_yscale('log')
        ax.legend()
        return ax
    
    def plotSpheroidSizeComparison(self,ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(self.absoluteMagSpheroid, self.Roman.data['spheroidRadius']*1e3, s=2, alpha=0.5, label="Galacticus")
        ax.scatter(self.MGCdataframe['ABS_MAG_GIM2DD'], self.MGCdataframe['RE_BULGE_KPC'],s=2,alpha=0.5, label="MGC")
        ax.set_ylim(1e-2, 1e2)
        ax.set_xlim(-23,-16)
        ax.set_xlabel(r'$M_B^\mathrm{spheroid}$')
        ax.set_ylabel(r'$R_\mathrm{spheroid} \, / \, \mathrm{kpc}$')
        ax.set_yscale('log')
        ax.legend()
        return ax
    
    def plotMGC_morphologyComparison(self, outputFname=None, title=None):
        fig, axs = plt.subplots(nrows=3, figsize=(6,10))
        self.plotBulgeToTotalComparison(axs[0])
        self.plotDiskSizeComparison(axs[1])
        self.plotSpheroidSizeComparison(axs[2])
        if title is not None:
            fig.suptitle(title)
        plt.tight_layout()
        if outputFname is not None:
            fig.savefig(outputFname)


    def attenuation_vs_inclination_comparison(self,diskMagRange=[-23,-17], spheroidMagRange=[-23,-17], BT_edges = [0,0.2,0.5,0.8,1.01], ylim=[-0.49,2.0], outputFname=None, title=None, figsize=None):
        BTbins = len(BT_edges)-1
        BT = self.bulgeToTotal
        if figsize is None:
            figsize = (8,3*BTbins)
        fig, axs = plt.subplots(nrows=BTbins,ncols=2,figsize=figsize)
        axs = np.atleast_2d(axs)
        inclinationBins = 10
        self.meanAttenDisk = np.zeros((BTbins,inclinationBins)); self.meanAttenSpheroid = np.zeros((BTbins,inclinationBins))
        self.meanAttenDiskErr = np.zeros((BTbins,inclinationBins)); self.meanAttenSpheroidErr = np.zeros((BTbins,inclinationBins))
        for i in np.arange(BTbins):
            # disk
            ax = axs[i,1]
            incut = (BT_edges[i] <= BT)*(BT < BT_edges[i+1])*(self.absoluteMagDisk < diskMagRange[1])*(self.absoluteMagDisk > diskMagRange[0])
            ax.scatter(1-np.cos(np.deg2rad(self.inclinations))[incut], self.diskAttenBbandMag[incut], alpha=0.5,s=2, label='Galacticus galaxies')
            x,y,yerr = get_binned_mean(1-np.cos(np.deg2rad(self.inclinations))[incut], self.diskAttenBbandMag[incut],xlim=[0,1],bins=inclinationBins)
            self.meanAttenDisk[i] = y; self.meanAttenDiskErr[i] = yerr
            ax.errorbar(x,y,yerr, c='r',marker='x', label='Galacticus mean')
            ax.set_yticks([])
            axx = ax.twinx()
            axx.set_yticks([])
            axx.set_ylabel(r"{} $ \leq B/T <$ {}".format(BT_edges[i], BT_edges[i+1]))
            # bulge
            ax = axs[i,0]
            incut = (BT_edges[i] <= BT)*(BT < BT_edges[i+1])*(self.absoluteMagSpheroid < spheroidMagRange[1])*(self.absoluteMagSpheroid > spheroidMagRange[0])
            ax.scatter(1-np.cos(np.deg2rad(self.inclinations))[incut], self.spheroidAttenBbandMag[incut], alpha=0.5,s=2, label='Galacticus galaxies')
            x,y,yerr = get_binned_mean(1-np.cos(np.deg2rad(self.inclinations))[incut], self.spheroidAttenBbandMag[incut],xlim=[0,1],bins=inclinationBins)
            self.meanAttenSpheroid[i] = y; self.meanAttenSpheroidErr[i] = yerr
            ax.errorbar(x,y,yerr, c='r',marker='x', label='Galacticus mean')

        axs[0,1].set_title("Discs")
        axs[0,0].set_title("Bulges")
        axs[BTbins//2,0].set_ylabel(r"$B$-band atten. (mag)")
        for ax in axs[BTbins-1]:
            ax.set_xlabel(r"$1 - \cos i$")
        for ax in axs.flatten():
            ax.set_xlim(0,0.99)
            ax.set_ylim(ylim)
            ax.axhline(0,ls='--',c='k')
        if title is not None:
            fig.suptitle(title)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0,wspace=0)
        if outputFname is not None:
            fig.savefig(outputFname)
        return axs
    
    def d07_Fig1_comparison(self, outputFname=None, faceOnAttenBulge=0.84, faceOnAttenDisk=0.20, **kwargs):
        axs = self.attenuation_vs_inclination_comparison( BT_edges = [0,0.8], **kwargs)
        axs[0,0].errorbar(d07_Fig1_MbStarCosI_Bulge['x'], faceOnAttenBulge+d07_Fig1_MbStarCosI_Bulge['y']-d07_Fig1_MbStarCosI_Bulge['y'][0], d07_Fig1_MbStarCosI_Bulge['yerr'], color='k', marker='s', ls='', markersize=4, label='Driver+ 07')
        axs[0,1].errorbar(d07_Fig1_MbStarCosI_Disc['x'], faceOnAttenDisk+d07_Fig1_MbStarCosI_Disc['y']-d07_Fig1_MbStarCosI_Disc['y'][0], d07_Fig1_MbStarCosI_Disc['yerr'], color='k', marker='s', ls='', markersize=4, label='Driver+ 07')
        if outputFname is not None:
            plt.savefig(outputFname)
        return axs
    
    def d07_Fig1_chi2(self, faceOnAttenDisk=0.20, faceOnAttenBulge=0.84, **kwargs):
        axs = self.attenuation_vs_inclination_comparison( BT_edges = [0,0.8], **kwargs)
        plt.close()
        chi2Disk = np.sum((faceOnAttenDisk+d07_Fig1_MbStarCosI_Disc['y']-d07_Fig1_MbStarCosI_Disc['y'][0] - self.meanAttenDisk)**2 / (d07_Fig1_MbStarCosI_Disc['yerr']**2 + self.meanAttenDiskErr**2))
        chi2Spheroid = np.sum((faceOnAttenBulge+d07_Fig1_MbStarCosI_Bulge['y']-d07_Fig1_MbStarCosI_Bulge['y'][0] - self.meanAttenSpheroid)**2 / (d07_Fig1_MbStarCosI_Bulge['yerr']**2 + self.meanAttenSpheroidErr**2))
        return chi2Disk, chi2Spheroid
    
    def plot_disk_and_spheroid_sizes(self, outputFname=None, BTbinEdges=[0,0.2,0.5],MdiskRange=[-20.5,-19.5], xlim=[1e-2, 1e2], ylim=[3e-4, 2e1]):
        BT = self.bulgeToTotal
        Mdisk = self.absoluteMagDisk
        Rdisk = self.Roman.data['diskRadius']*1e3 
        Rspheroid = self.Roman.data['spheroidRadius']*1e3
        BTbins = len(BTbinEdges)-1
        plt.figure()
        for i in np.arange(BTbins):
            include = (BT>BTbinEdges[i])*(BT<BTbinEdges[i+1])*(Mdisk<MdiskRange[1])*(Mdisk>MdiskRange[0])
            plt.scatter(Rdisk[include], Rspheroid[include], alpha=0.5, s=2, label=r'${} < B/T < {}$'.format(BTbinEdges[i], BTbinEdges[i+1]))
        plt.xlabel('Rdisk / kpc')
        plt.ylabel('Rspheroid / kpc')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.xlim(xlim)
        plt.ylim(ylim)
        if outputFname is not None:
            plt.savefig(outputFname)





    def plot_demonstrating_d07_method(self, component='disk', cos_inclination_edges = np.linspace(0,1,11), BT_edges = np.array([0, 0.2,]), meanAttenuationWindowWidth=0.5, maglims=[-23,-16], phi_ylim=[0.1,3e4], fixed_alpha=-1.25):
        ''' Make a plot demonstrating how the shift in M_B* from a Schechter function fit
        can be used to infer a typical B-band attenuation '''
        if component=='disk':
            magNoDust = self.absoluteMagDisk
            magDust = self.absoluteMagDisk + self.diskAttenBbandMag
        elif component=='spheroid':
            magNoDust = self.absoluteMagSpheroid
            magDust = self.absoluteMagSpheroid + self.spheroidAttenBbandMag
        else:
            raise ValueError("component must be one of 'disk' or 'spheroid'")

        # new figure for each BT-bin
        for j in np.arange(len(BT_edges)-1):
            fig, axs = plt.subplots(ncols=2, nrows=len(cos_inclination_edges)-1,figsize=(10,4*(len(cos_inclination_edges)-1)))
            # new row for each inclination bin
            for i in np.arange(len(cos_inclination_edges)-1):
                # find the relevant sample of galaxies
                samp = (self.cos_inclinations > cos_inclination_edges[i]) * (self.cos_inclinations < cos_inclination_edges[i+1]) * (self.bulgeToTotal < BT_edges[j+1]) * (self.bulgeToTotal >= BT_edges[j])
                # and measure and fit their "dust" and "no dust" Schechter functions
                schechterNoDust = luminosity_function(magNoDust[samp], fixed_alpha=fixed_alpha)
                schechterDust = luminosity_function(magDust[samp], fixed_alpha=fixed_alpha)
                # Calculate running mean attenuation with a window width of 0.5 mags
                x_mean, y_mean = running_mean_bin(magNoDust[samp], (magDust-magNoDust)[samp], window_width=meanAttenuationWindowWidth, xs=np.linspace(*maglims,100))
                # Make the M_B vs A_B plot
                ax = axs[i][0]
                ax.scatter(magNoDust[samp], (magDust-magNoDust)[samp], label="individual galaxies", alpha=0.2,s=2)
                ax.plot(x_mean, y_mean,color='red', label="running mean", zorder=100)
                inferred_A_B = schechterDust.schechter_params[1] - schechterNoDust.schechter_params[1]
                inferred_A_Berr = np.sqrt(schechterDust.schechter_cov[1,1] + schechterNoDust.schechter_cov[1,1])
                ax.axhline(inferred_A_B, ls='--',c='b', label='inferred $A_B$')
                ax.axhspan(inferred_A_B-inferred_A_Berr, inferred_A_B+inferred_A_Berr, color='b',alpha=0.2)
                ax.axvline(schechterNoDust.schechter_params[1], label='$M_B^*$',c='k',ls='--')
                ax.axhline(0, ls='-', c='k')
                ax.set_xlim(-23,-16)
                #plt.yscale('log')
                ax.set_ylim(-0.5,2) #plt.ylim(1e-3,10)
                ax.set_xlabel(r'$M_B$')
                ax.set_ylabel(r'$A_B$')
                if i==0:
                    ax.legend(loc='upper right', fontsize='small')
                ax.axvspan(schechterNoDust.range[1], 0, color='k', alpha=0.5) # faint limit of Schechter fit
                # Make the luminosity function plot
                ax = axs[i][1]
                schechterNoDust.plot_results(label='no dust', ax=ax)
                schechterDust.plot_results(ax=ax, color='red', label='dust',shift=0.04)
                ax.axvline(schechterNoDust.schechter_params[1], label='$M_B^*$',c='k',ls='--')
                ax.axvline(schechterDust.schechter_params[1], c='r',ls='--')
                ax.set_xlim(maglims)
                ax.set_ylim(phi_ylim)
                # include uncertainty
                mags = np.linspace(*maglims,100)
                # without dust
                schecter_param_samples = np.random.multivariate_normal(schechterNoDust.schechter_params, schechterNoDust.schechter_cov, 100)
                for schecter_params in schecter_param_samples:
                    ax.plot(mags, schechter_function(mags, *schecter_params), color='k', alpha=0.05, zorder=np.random.uniform(0,1))
                # with dust
                schecter_param_samples = np.random.multivariate_normal(schechterDust.schechter_params, schechterDust.schechter_cov, 100)
                for schecter_params in schecter_param_samples:
                    ax.plot(mags, schechter_function(mags, *schecter_params), color='red', alpha=0.05, zorder=np.random.uniform(0,1))
                if i==0:
                    ax.legend(fontsize='small')
                ax.axvspan(schechterNoDust.range[1], 0, color='k', alpha=0.5) # faint limit of Schechter fit



d07_Fig1_MbStarCosI_Bulge = {
    'x': np.linspace(0.05,0.95,10),
    'y': np.array([-19.00, -19.00, -18.93, -18.94, -18.82, -18.62, -17.91, -18.06, -18.01, -16.7]), # last point off the plot, so a bit of a guess, have inflated the errorbar
    'yerr': np.array([0.1 , 0.08, 0.07, 0.07, 0.1 , 0.11, 0.11, 0.14, 0.16, 0.3]),
}

d07_Fig1_MbStarCosI_Disc = {
    'x': np.linspace(0.05,0.95,10),
    'y': np.array([-19.378, -19.369, -19.322, -19.235, -19.218, -19.166, -19.049, -18.819, -18.693, -18.593]),
    'yerr': np.array([0.058, 0.049, 0.052, 0.065, 0.058, 0.066, 0.069, 0.069, 0.063, 0.133]),
}

d07_Fig7_correctedBulge = {
    'x': np.array([-21.749, -21.253, -20.752, -20.239, -19.749, -19.242, -18.740, -18.245, -17.743, -17.242]),
    'y': np.array([-5.334, -4.847, -3.856, -3.291, -3.002, -2.841, -2.751, -2.746, -2.722, -2.982]),
}


d07_Fig7_correctedDisc = {
    'x': np.array([-21.751, -21.247, -20.743, -20.240, -19.737, -19.239, -18.737, -18.235, -17.738, -17.230]),
    'y': np.array([-4.865, -4.094, -3.229, -2.861, -2.522, -2.298, -2.180, -2.041, -1.980, -1.980]),
}






