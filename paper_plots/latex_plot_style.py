# latex_plot_style.py

import matplotlib.pyplot as plt
import matplotlib as mpl

def set_latex_style():
    mpl.rcParams.update({
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'axes.labelsize': 16,
        'legend.fontsize': 12,
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'axes.unicode_minus': False,
    })