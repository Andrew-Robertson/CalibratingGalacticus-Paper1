from scipy.special import gamma
from astropy.cosmology import Planck15
import astropy.units as u
import astropy.constants as c
import numpy as np
import h5py
import matplotlib.pyplot as plt

# comparison with paper: https://ui.adsabs.harvard.edu/abs/2007MNRAS.376..215B/abstract

def BettSpinDistribution(loglambda, alpha=2.509, l0=0.04326):
    l = 10**loglambda
    A = 3*np.log(10) * alpha**(alpha-1) / gamma(alpha)
    return A*(l/l0)**3*np.exp(-alpha * (l/l0)**(3./alpha))

def Lvir(Mvir):
    # calculate the angular momentum of a particle on a circular orbit at Rvir (multiplied by sqrt(2))
    Rvir = (Mvir / (1.333*np.pi*200*Planck15.critical_density0))**(1./3)
    Vvir = np.sqrt(c.G*Mvir/Rvir)
    return np.sqrt(2)*(Mvir * Vvir * Rvir).to('Msun Mpc km / s')

def calculateGalacticusSpins(nodeData, selection=None):
    # nodeData is a Galacticus hdf5 Output group's nodeData
    if selection is None:
        # Default selection that includes all data
        selection = lambda g: np.ones(len(g['basicMass']), dtype=bool)
    sel = selection(nodeData)
    lambdaBullock = (nodeData['spinAngularMomentum'][sel] / Lvir(nodeData['basicMass'][sel]*u.Msun)).value # Bullock spin param
    return lambdaBullock

def basicMassSelection(g,Mmin=1e12):
    # Select nodes where the total mass is greater than Mmin Msun
    return g['basicMass'][:]> Mmin

def spinComparisonPlot(fname, OutputGroup=1, xlim=[-3,0.5], outputFname=None, hasSpin=False):
    # fname is path to a galacticus *.hdf5 file
    with h5py.File(fname) as f:
        g = f['Outputs/Output'+str(OutputGroup)+'/nodeData']
        plt.figure()        
        if hasSpin:
            allLambda = g['spinParameter'][:]
            highMassLambda = g['spinParameter'][:][basicMassSelection(g)]
        else:
            allLambda = calculateGalacticusSpins(g)
            highMassLambda = calculateGalacticusSpins(g, selection=basicMassSelection)
    plt.hist(np.log10(allLambda),density=True,bins=60, range=xlim, label='galacticus all halos',alpha=0.5)     
    plt.hist(np.log10(highMassLambda),density=True,bins=60, range=xlim, label='galacticus basicMass>1e12', alpha=0.5)
    logLambdas = np.linspace(*xlim,100)
    plt.plot(logLambdas, BettSpinDistribution(logLambdas), label='Bett+ 2007')
    plt.legend(fontsize='small')
    plt.xlabel(r"$\log_{10} \lambda$")
    plt.yticks([])
    plt.xlim(xlim)
    if outputFname is not None:
        plt.savefig(outputFname)
        

    
