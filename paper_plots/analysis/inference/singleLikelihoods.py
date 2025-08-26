import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml
import time
import subprocess
import re
from lxml import etree as ET  # Use lxml for proper parent tracking

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

def load_config(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config

class SHMR_likelihood():
    ''' A class for reading in and visualising a likelihood that comes from
    trying to match the stellar mass - halo mass relation at two reshifts in
    two different halo mass bins.'''

    def __init__(self, path, runAll=False, lowzFname="romanEPS_SHMR_lowz.hdf5", highzFname="romanEPS_SHMR_highz.hdf5"):
        self.path = path
        self.lowzFname = lowzFname
        self.highzFname = highzFname
        if runAll:
            self.load_data()
            self.make4plots()

    def load_data(self,):
        if self.lowzFname is not None:
            with h5py.File(os.path.join(self.path, self.lowzFname)) as f:
                self.data_lowz = h5Obj_to_dict(f)
            self.SHMR_lowz = self.data_lowz['analyses']['stellarHaloMassRelationLeauthaud2012z1']
            self.SHMRscatter_lowz = self.data_lowz['analyses']['stellarHaloMassRelationScatterLeauthaud2012z1']
        else:
            self.data_lowz = None
            self.SHMR_lowz = None
            self.SHMRscatter_lowz = None
        if self.highzFname is not None:
            with h5py.File(os.path.join(self.path, self.highzFname)) as f:
                self.data_highz = h5Obj_to_dict(f)
            self.SHMR_highz = self.data_highz['analyses']['stellarHaloMassRelationLeauthaud2012z3']
            self.SHMRscatter_highz = self.data_highz['analyses']['stellarHaloMassRelationScatterLeauthaud2012z3']
        else:
            self.data_highz = None
            self.SHMR_highz = None
            self.SHMRscatter_highz = None

    def plot_SHMR(self, SHMRdata, ax=None, plot_target=True, color='C0', offset_text=0, model_label='Galacticus', xshift=0, print_chi2=True):
        if SHMRdata is None:
            return
        if ax is None:
            fig, ax = plt.subplots()
        inc = SHMRdata['massStellarLog10Target'] != 0
        if plot_target:
            ax.errorbar(np.log10(SHMRdata['massHalo'])[inc], SHMRdata['massStellarLog10Target'][inc],
                        np.sqrt(np.diag(SHMRdata['massStellarLog10CovarianceTarget']))[inc],
                        label='Target', ls='', marker='o', color='C1')
        ax.errorbar(np.log10(SHMRdata['massHalo'])[inc]+xshift, SHMRdata['massStellarLog10'][inc],
                    np.sqrt(np.diag(SHMRdata['massStellarLog10Covariance']))[inc],
                    label=model_label, ls='', marker='o', color=color)
        ax.set_ylim(9, 12)
        ax.set_xlabel('log10 massHalo')
        ax.set_ylabel('log10 massStellar')
        if print_chi2:
            dy = 0.08  # one line of vertical space in axes coords
            ax.text(x=0.68, y=0.05 + offset_text * dy,
                s=r"$\chi^2$ = {:.1f}".format(-2 * SHMRdata['attributes']['logLikelihood']),
                fontsize=12, transform=ax.transAxes, horizontalalignment='left', color=color)


    def plot_SHMRscatter(self, SHMRscatterData, ax=None, plot_target=True, color='C0', offset_text=0, model_label='Galacticus', xshift=0, print_chi2=True):
        if SHMRscatterData is None:
            return
        if ax is None:
            fig, ax = plt.subplots()
        inc = SHMRscatterData['massStellarLog10ScatterTarget'] != 0
        if plot_target:
            ax.errorbar(np.log10(SHMRscatterData['massHalo'])[inc], SHMRscatterData['massStellarLog10ScatterTarget'][inc],
                        np.sqrt(np.diag(SHMRscatterData['massStellarLog10ScatterCovarianceTarget']))[inc],
                        label='Target', ls='', marker='o', color='C1')
        ax.errorbar(np.log10(SHMRscatterData['massHalo'])[inc]+xshift, SHMRscatterData['massStellarLog10Scatter'][inc],
                    np.sqrt(np.diag(SHMRscatterData['massStellarLog10ScatterCovariance']))[inc],
                    label=model_label, ls='', marker='o', color=color)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel('log10 massHalo')
        ax.set_ylabel('scatter in log10 massStellar')
        if print_chi2:
            dy = 0.08  # one line of vertical space in axes coords
            ax.text(x=0.68, y=0.95 - offset_text * dy,
                    s=r"$\chi^2$ = {:.1f}".format(-2 * SHMRscatterData['attributes']['logLikelihood']),
                    fontsize=12, transform=ax.transAxes, horizontalalignment='left', verticalalignment='top', color=color)

    def make4plots(self, axs=None, fname=None, **kwargs):
        if axs is None:
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
            fig_created = True
        else:
            fig = axs[0, 0].figure
            fig_created = False

        self.plot_SHMR(self.SHMR_lowz, ax=axs[0, 0], **kwargs)
        self.plot_SHMR(self.SHMR_highz, ax=axs[0, 1], **kwargs)
        self.plot_SHMRscatter(self.SHMRscatter_lowz, ax=axs[1, 0], **kwargs)
        self.plot_SHMRscatter(self.SHMRscatter_highz, ax=axs[1, 1], **kwargs)

        axs[0, 0].set_title('z = 0.32')
        axs[0, 1].set_title('z = 0.88')
        axs[1, 1].legend()

        if fig_created:
            runTime = self.data_lowz['Version']['attributes']['runDuration']
            odeTol = self.data_lowz['Parameters']['mergerTreeNodeEvolver']['attributes']['odeToleranceRelative']
            plt.suptitle("odeTol={}\nLow-z Run Time = {:.1f} s".format(odeTol, runTime), verticalalignment='top')
            plt.tight_layout()

        if fname is not None:
            fig.savefig(fname)

        return fig, axs




# Shen 2003 Relations
def shen2003_meanEarlyTypeRadiusOriginal(M, b=2.88e-6, a=0.56):
    # result in kpc
    return b * M**a

Mp = 1e10
bpShen = 2.88e-6 * (Mp)**0.56
gammapShen = 0.1 * (Mp)**0.14
def shen2003_meanEarlyTypeRadius(M, bp=bpShen, a=0.56):
    # result in kpc
    return bp * (M/Mp)**a

def shen2003_meanLateTypeRadius(M, alpha=0.14, beta=0.39, massZeroPoint=3.98e10, gammap=gammapShen):
    # result in kpc
    return gammap * (M/Mp)**alpha * (1+M/massZeroPoint)**(beta-alpha)

def shen2003_sigmaLogR(M, sigma1=0.47, sigma2=0.34, massZeroPoint=3.98e10):
    return sigma2 + (sigma1 - sigma2)/(1 + (M/massZeroPoint)**2)


def mean_with_errors(M, R, sigmaLogR):
    low = R * np.exp(-sigmaLogR)
    med = R
    high = R * np.exp(sigmaLogR)
    return low, med, high

def legend_in_top_left(legend, fig, xmax=0.3, ymin=0.5):
    # Get bounding box in Axes coordinates (normalized between 0 and 1)
    fig.canvas.draw()  # Ensure everything is rendered
    bbox = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    # Extract legend's lower-left corner
    return bbox.x0 < xmax and bbox.y1 > ymin


coloursShen = {'disk': 'C0', 'spheroid': 'C3'}

class mass_size_likelihood():
    ''' A class for reading in and visualising a likelihood that comes from
    trying to match the stellar mass - size relation at low-reshift'''

    def __init__(self, path, runAll=False, fname="romanEPS_SHMR_lowz.hdf5"):
        self.path = path
        self.fname = fname
        if runAll:
            self.load_data()
            _ = self.noBinLogLikelihood()
            self.plotMstarRstar()

    def load_data(self,):
        with h5py.File(os.path.join(self.path, self.fname)) as f:
            self.data = h5Obj_to_dict(f)
        try:
            self.galacticusLogLikelihood = self.data['analyses']['massSizeRelationShen2003']['attributes']['logLikelihood']
        except KeyError:
            print("No massSizeRelationShen2003 analysis found in the file.")
            self.galacticusLogLikelihood = None
        self.nodeData = self.data['Outputs']['Output1']['nodeData']
        self.central = self.nodeData['nodeIsIsolated'][:].astype('bool')
        self.r50 = 1e3*self.nodeData['radiusHalfMassStellar'][self.central]
        self.Mstar = self.nodeData['diskMassStellar'][self.central]+self.nodeData['spheroidMassStellar'][self.central]
        self.BT = self.nodeData['spheroidMassStellar'][self.central]/self.Mstar
        self.earlyType = (self.BT>0.5)
        self.lateType = ~self.earlyType

    def plotMstarRstar(self, Ms = np.geomspace(3e8,1e12,100), xlim=[3e8,1e12], ylim=[0.07,300], ax=None, fname=None, write_chi2=True):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,4))
        else:
            fig = ax.figure
        low, med, high = mean_with_errors(Ms, shen2003_meanEarlyTypeRadius(Ms), shen2003_sigmaLogR(Ms))
        ax.fill_between(Ms, low, high, alpha=0.5, color=coloursShen['spheroid'])
        ax.plot(Ms, shen2003_meanEarlyTypeRadius(Ms), label='SDSS early-type', color=coloursShen['spheroid'])
        low, med, high = mean_with_errors(Ms, shen2003_meanLateTypeRadius(Ms), shen2003_sigmaLogR(Ms))
        ax.fill_between(Ms, low, high, alpha=0.5, color=coloursShen['disk'])
        ax.plot(Ms, shen2003_meanLateTypeRadius(Ms), label='SDSS late-type', color=coloursShen['disk'])
        ax.scatter(self.Mstar[self.earlyType], self.r50[self.earlyType], label='Galacticus bulge-dominated', color=coloursShen['spheroid'])
        ax.scatter(self.Mstar[~self.earlyType], self.r50[~self.earlyType], label='Galacticus disk-dominated', color=coloursShen['disk'])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$M_\star \, / \, \mathrm{M_\odot}$')
        ax.set_ylabel(r'$R_\star^{50} \, / \, \mathrm{kpc}$')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        legend = ax.legend()
        if write_chi2:
            if legend_in_top_left(legend, fig):
                ax.text(0.99, 0.01, r'$\chi^2 = {:.1f}$'.format(self.noBinChi2), horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
            else:
                ax.text(0.01, 0.99, r'$\chi^2 = {:.1f}$'.format(self.noBinChi2), horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
        plt.tight_layout()
        if fname is not None:
            plt.savefig(fname)
        return ax

    def noBinLogLikelihood(self,):
        chi2contributionEarlyType = np.log(self.r50[self.earlyType] / shen2003_meanEarlyTypeRadius(self.Mstar[self.earlyType]))**2 / shen2003_sigmaLogR(self.Mstar[self.earlyType])**2
        chi2contributionLateType = np.log(self.r50[self.lateType] / shen2003_meanLateTypeRadius(self.Mstar[self.lateType]))**2 / shen2003_sigmaLogR(self.Mstar[self.lateType])**2
        self.noBinChi2 = np.sum(chi2contributionEarlyType) + np.sum(chi2contributionLateType)
        return -0.5*self.noBinChi2 + np.log(np.prod(1/(np.sqrt(2*np.pi)*shen2003_sigmaLogR(self.Mstar))))


# Function to read the XML template from a file
def read_xml_template(template_file):
    with open(template_file, 'r') as file:
        return file.read()
    
defaultParameterValues = {
        "outputFileName": "defaultOutputFilename.hdf5",
        "replicationCount": 9,
        "starFormationRateSpheroids_efficiency": 0.04,
        "starFormationRateSpheroids_exponentVelocity": 2.0,
        "stellarFeedbackDisks_exponent": 2.70,
        "stellarFeedbackDisks_velocityCharacteristic": 165.0,
        "stellarFeedbackSpheroids_exponent": 1.80,
        "stellarFeedbackSpheroids_velocityCharacteristic": 80.0,
        "randomNumberGenerator_seed":219,
        "satelliteMergingRadiusTrigger_radiusVirialFraction": 0.01,
        "massResolution": 2.0e10,
        "massResolutionFractional": 0.001,
    }

def generate_xml_content(template,  **kwargs):
    # Merge defaults with provided kwargs (later things overwrite earlier ones)
    context = {**defaultParameterValues, **kwargs}
    # Format the template
    xml_content = template.format(**context)
    return xml_content

# Function to write the XML content to a file
def write_xml_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)

def convert_bool_kwargs(**kwargs):
    converted_kwargs = {
        key: "true" if value is True else "false" if value is False else value
        for key, value in kwargs.items()
    }
    return converted_kwargs

def modify_xml(xml_file, modifications, replacements, output_file=None):
    """
    Reads an XML file, modifies specified values, replaces elements, and optionally writes back.

    Parameters:
    - xml_file (str): Path to the input XML file.
    - modifications (dict): A dictionary where keys are XPath expressions and values are replacements.
    - replacements (dict): A dictionary where keys are XPath expressions and values are lists of new XML elements.
    - output_file (str, optional): Path to save the modified XML file.

    Returns:
    - ET.ElementTree: Modified XML as an ElementTree object.
    """
    # Parse the XML file
    parser = ET.XMLParser(remove_blank_text=True)  # Preserve formatting
    tree = ET.parse(xml_file, parser)
    root = tree.getroot()

    # Modify specified values
    for xpath, new_value in modifications.items():
        for elem in root.xpath(xpath):
            elem.set("value", new_value)

    # Replace elements with new structures
    for xpath, new_elements in replacements.items():
        for elem in root.xpath(xpath):
            parent = elem.getparent()  # This works in lxml!
            if parent is not None:
                parent.remove(elem)  # Remove old element
                for new_elem in new_elements:
                    parent.append(new_elem)  # Add new elements

    # Save modified XML if an output file is provided
    if output_file:
        tree.write(output_file, encoding="utf-8", xml_declaration=True, pretty_print=True)

    return tree  # Return the modified tree for further use

class galacticusRun():

    def __init__(self, templateFname="/home/arobertson/Galacticus/noodling/Parameter_optimisation/constrainingGalacticusParameters/MCMC/tutorial/singleLikelihood/templateParameterFiles/romanEPS_SHMR_lowz.xml",
                    ):
        self.templateFname = templateFname
        self.template = read_xml_template(self.templateFname)
        
    def generateParameterFile(self,
                        galacticusParameterFileName='likelihoodEval.xml', 
                        galacticusOutputFileName="likelihoodEval.hdf5",
                        **parameterUpdates):
        # parameterUpdates are values from the parameter file template that are to be set
        self.galacticusParameterFileName = galacticusParameterFileName
        self.galacticusOutputFileName = galacticusOutputFileName
        self.xml_content = generate_xml_content(self.template, outputFileName=self.galacticusOutputFileName, **convert_bool_kwargs(**parameterUpdates))
        write_xml_file(self.galacticusParameterFileName, self.xml_content)

    def generateParameterFileElementTree(self,
                        galacticusParameterFileName='likelihoodEval.xml', 
                        galacticusOutputFileName="likelihoodEval.hdf5",
                        modifications={}, replacements={}):
        self.galacticusParameterFileName = galacticusParameterFileName
        self.galacticusOutputFileName = galacticusOutputFileName
        modifications[".//outputFileName"] = galacticusOutputFileName
        modify_xml(self.templateFname, modifications, replacements, output_file=self.galacticusParameterFileName)


    def runGalacticus(self, fname=None, silent=True, executableName="Galacticus.exeZeroNegativeMasses", submitSLURM=False, cpus=9, memPerCPU=4):
        if fname is None:
            fname = self.galacticusParameterFileName

        command = f"$GALACTICUS_EXEC_PATH/{executableName} {fname}"
        if silent:
            command +=  "> /dev/null 2>&1"

        if submitSLURM:
            sbatch_command = f"sbatch --cpus-per-task={cpus} --export=ALL,OMP_NUM_THREADS={cpus} --mem-per-cpu={memPerCPU}G --wrap='{command}'"
            
            # Submit SLURM job and capture job ID
            result = subprocess.run(sbatch_command, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"SLURM submission failed: {result.stderr}")
            
            # Extract job ID from sbatch output
            match = re.search(r'Submitted batch job (\d+)', result.stdout)
            if not match:
                raise RuntimeError(f"Failed to parse SLURM job ID from output: {result.stdout}")
            
            job_id = match.group(1)
            print(f"Submitted SLURM job {job_id}. Waiting for completion...")

            # Wait for the job to finish
            while True:
                squeue_result = subprocess.run(f"squeue -j {job_id}", shell=True, capture_output=True, text=True)
                if job_id not in squeue_result.stdout:
                    break  # Job is no longer in the queue, so it's done
                time.sleep(3)  # Wait a bit before checking again

            print(f"SLURM job {job_id} completed.")
        else:
            exit_code = os.system(command)
            if exit_code != 0:
                raise RuntimeError(f"Command failed with exit code {exit_code}")
        
class SHMR_run():

    def __init__(self, templateFnames=["/home/arobertson/Galacticus/noodling/Parameter_optimisation/constrainingGalacticusParameters/MCMC/tutorial/singleLikelihood/templateParameterFiles/romanEPS_SHMR_lowz.xml",
                                       "/home/arobertson/Galacticus/noodling/Parameter_optimisation/constrainingGalacticusParameters/MCMC/tutorial/singleLikelihood/templateParameterFiles/romanEPS_SHMR_highz.xml"],
            executableName="Galacticus.exeZeroNegativeMasses",
            outputDir='exampleOutput',
            updatedParameters={}, silent=False, submitSLURM=False, cpus=9):
        
        self.templateFnames = templateFnames
        self.executableName = executableName
        self.outputDir = outputDir
        self.updatedParameters = updatedParameters

        os.makedirs(self.outputDir, exist_ok=True)

        for templateFname in self.templateFnames:
            galRun = galacticusRun(templateFname)
            galRun.generateParameterFile(galacticusParameterFileName=os.path.join(self.outputDir, os.path.basename(templateFname)),
                                         galacticusOutputFileName=os.path.join(self.outputDir, os.path.basename(templateFname).replace(".xml", ".hdf5")),
                **self.updatedParameters)
            galRun.runGalacticus(silent=silent, executableName=self.executableName, submitSLURM=submitSLURM, cpus=cpus)
        SHMR_likelihood(self.outputDir, runAll=True)

        
