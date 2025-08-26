import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import pandas as pd
from latex_plot_style import set_latex_style
import re
from analysis.inference import singleLikelihoods as sl
import analysis.data_comparison.galacticusAutoAnalysis as analysis
from analysis.inference import mcmc

try:
    # If you want a specific cosmology, pass it in; otherwise this gives you Planck18.
    from astropy.cosmology import Planck18 as _DEFAULT_COSMO
    _HAS_ASTROPY = True
except Exception:
    _DEFAULT_COSMO = None
    _HAS_ASTROPY = False


def speagle14_log_sfr_ms(log10_Mstar, t_gyr=None, z=None, cosmology=None):
    """
    Speagle+ (2014) star-forming main sequence (SFMS) meta-fit.

    Parameters
    ----------
    log10_Mstar : array_like
        log10 stellar mass [M_sun].
    t_gyr : float or array_like, optional
        Age of the Universe at the galaxy redshift [Gyr]. Provide this OR `z`.
    z : float or array_like, optional
        Redshift(s). If given, `t_gyr` is computed from the provided `cosmology` (or Planck18).
    cosmology : astropy.cosmology instance, optional
        Used only if `z` is provided. Defaults to Planck18 if available.

    Returns
    -------
    log10_SFR_ms : ndarray
        log10 SFR on the main sequence [M_sun/yr].
    """
    log10_Mstar = np.asanyarray(log10_Mstar, dtype=float)

    if t_gyr is None:
        if z is None:
            raise ValueError("Provide either t_gyr or z.")
        if cosmology is None:
            if not _HAS_ASTROPY:
                raise ImportError("astropy not available. Provide `t_gyr` directly or install astropy.")
            cosmology = _DEFAULT_COSMO
        # Vectorized age of the Universe in Gyr
        t_gyr = np.asanyarray(cosmology.age(np.asanyarray(z)).value, dtype=float)
    else:
        t_gyr = np.asanyarray(t_gyr, dtype=float)

    # Speagle+14: log SFR = (0.84 - 0.026 t) * log M* - (6.51 - 0.11 t), with t in Gyr.
    a = 0.84 - 0.026 * t_gyr
    b = -(6.51 - 0.11 * t_gyr)
    return a * log10_Mstar + b


def classify_starforming_quiescent(log10_Mstar, log10_SFR,
                                   t_gyr=None, z=None, cosmology=None,
                                   delta_threshold=-1.0, return_delta=False):
    """
    Label galaxies as star-forming or quiescent by offset from the SFMS.

    Definition:
        Δ_MS = log10(SFR) - log10(SFR_MS(M*, z))
        quiescent if Δ_MS <= delta_threshold (default: -1.0 dex), else star-forming.

    Parameters
    ----------
    log10_Mstar : array_like
        log10 stellar mass [M_sun].
    log10_SFR : array_like
        log10 SFR [M_sun/yr].
    t_gyr, z, cosmology : see speagle14_log_sfr_ms
    delta_threshold : float, optional
        Threshold in dex below the SFMS to label as quiescent (default: -1.0).
    return_delta : bool, optional
        If True, also return Δ_MS array.

    Returns
    -------
    labels : ndarray of dtype '<U11'
        Array with values "quiescent" or "star-forming".
    delta_ms : ndarray, optional
        Δ_MS values (only if return_delta=True).
    """
    log10_Mstar = np.asanyarray(log10_Mstar, dtype=float)
    log10_SFR = np.asanyarray(log10_SFR, dtype=float)

    ms = speagle14_log_sfr_ms(log10_Mstar, t_gyr=t_gyr, z=z, cosmology=cosmology)
    delta_ms = log10_SFR - ms

    labels = np.where(delta_ms <= delta_threshold, "quiescent", "star-forming")
    return (labels, delta_ms) if return_delta else labels


def _find_observed_key(group, comp, band):
    """
    Find a dataset key like
      '{comp}LuminositiesStellar:SDSS_{band}:observed:z<xxxx>'
    and return (key, redshift_string), e.g. ('diskLuminositiesStellar:SDSS_u:observed:z0.3521','z0.3521').
    """
    prefix = f"{comp}LuminositiesStellar:SDSS_{band}:observed:z"
    for k in group.keys():
        if k.startswith(prefix):
            m = re.search(r":observed:(z[\d.]+)$", k)
            return k, (m.group(1) if m else None)
    raise KeyError(f"Could not find observed luminosity for {comp} {band} in group '{group.name}'.")

def _safe_mag(total_lum):
    """Return -2.5 log10(total_lum) with non-positive values set to nan."""
    total_lum = np.asarray(total_lum)
    out = np.full_like(total_lum, np.nan, dtype=float)
    m = total_lum > 0
    out[m] = -2.5*np.log10(total_lum[m])
    return out

def read_galacticus_output(fname, output, bands=("u","g","z")):
    """
    Read one /Outputs/Output<output>/nodeData block from a Galacticus HDF5 file,
    auto-detect the ':observed:zXXXX' suffix for requested SDSS bands, and return a dict.

    Parameters
    ----------
    fname : str
        Path to HDF5 file.
    output : int or str
        Output number (e.g., 16 or "16").
    bands : iterable of str
        Bands to load magnitudes for (SDSS names like 'u','g','r','i','z').

    Returns
    -------
    data : dict
        {
          'output': int,
          'redshift': float or None,
          'redshift_token': str (e.g. 'z0.3521') or None,
          'r50_kpc': array,
          'Mstar': array,
          'BT': array,
          'SFR_Msun_per_yr': array,
          'mags': {'u': array, 'g': array, 'z': array, ...}
        }
    """
    outnum = int(output)
    base = f"/Outputs/Output{outnum}"
    node = f"{base}/nodeData"

    with h5py.File(fname, "r") as f:
        g = f[node]

        # Core quantities (mirror your snippet)
        r50_kpc = g["radiusHalfMassStellar"][:]*1e3  # assume input is in Mpc
        Mdisk = g["diskMassStellar"][:]
        Mbulge = g["spheroidMassStellar"][:]
        Mstar = Mdisk + Mbulge
        # avoid divide-by-zero
        BT = np.divide(Mbulge, Mstar, out=np.zeros_like(Mbulge, dtype=float), where=Mstar>0)

        SFR = (g["diskStarFormationRate"][:] + g["spheroidStarFormationRate"][:]) / 1e9  # to Msun/yr

        # Auto-detect the redshift token once using the first band we find
        redshift_token = None
        mags = {}
        for b in bands:
            disk_key, ztok = _find_observed_key(g, "disk", b)
            sph_key, _    = _find_observed_key(g, "spheroid", b)
            if redshift_token is None and ztok is not None:
                redshift_token = ztok

            total_lum = g[disk_key][:] + g[sph_key][:]
            mags[b] = _safe_mag(total_lum)

        # Try to convert token -> numeric redshift if possible
        redshift = None
        if redshift_token:
            try:
                redshift = float(redshift_token[1:])  # strip leading 'z'
            except ValueError:
                pass

        # Fallback: some files store an explicit redshift dataset/attr
        if redshift is None:
            # try /Outputs/OutputX/redshift or attribute
            if f.get(f"{base}/redshift") is not None:
                redshift = np.array(f[f"{base}/redshift"][()]).astype(float).item()
            elif "redshift" in f[base].attrs:
                redshift = float(f[base].attrs["redshift"])

    return {
        "output": outnum,
        "redshift": redshift,
        "redshift_token": redshift_token,
        "r50_kpc": r50_kpc,
        "Mstar": Mstar,
        "BT": BT,
        "SFR_Msun_per_yr": SFR,
        "mags": mags,
    }

def running_percentiles(x, y, bins, percentiles=(16, 50, 84),
                        ax=None, plot=True, min_count=1,
                        line_kwargs=None, band_kwargs=None):
    """
    Compute (and optionally plot) running percentiles of y as a function of x.

    Parameters
    ----------
    x, y : array-like
        Data arrays.
    bins : array-like
        Bin edges in x (e.g., np.logspace for log bins).
    percentiles : iterable of ints/floats, optional
        Percentiles to compute (default: (16, 50, 84)).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None and plot=True, uses current axes.
    plot : bool, optional
        If True, plot results (median line if 50 is requested; shaded
        band if both 16 and 84 are requested).
    min_count : int, optional
        Minimum number of points required in a bin to report stats.
    line_kwargs : dict, optional
        Passed to ax.plot() for the median (or percentile) line.
        Defaults: {'color':'k', 'lw':2}.
    band_kwargs : dict, optional
        Passed to ax.fill_between() for the 16–84% band.
        Defaults: {'alpha':0.25, 'lw':0}.

    Returns
    -------
    bin_centers : ndarray
        Geometric mean of bin edges (len = len(bins)-1).
    pct : dict
        Mapping {p: array} with one array per requested percentile.
        Each array has length len(bins)-1 and NaN where count < min_count.
    counts : ndarray
        Number of points in each bin.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]

    percentiles = tuple(percentiles)
    nb = len(bins) - 1
    bin_centers = np.sqrt(bins[:-1] * bins[1:])

    pct = {p: np.full(nb, np.nan, dtype=float) for p in percentiles}
    counts = np.zeros(nb, dtype=int)

    # Left-inclusive, right-exclusive (last bin right-inclusive)
    for i in range(nb):
        if i < nb - 1:
            sel = (x >= bins[i]) & (x < bins[i+1])
        else:
            sel = (x >= bins[i]) & (x <= bins[i+1])

        yi = y[sel]
        counts[i] = yi.size
        if counts[i] >= min_count:
            for p in percentiles:
                pct[p][i] = np.percentile(yi, p)

    if plot:
        if ax is None:
            ax = plt.gca()

        # Shaded band if 16 & 84 present
        if 16 in pct and 84 in pct:
            bk = {'alpha': 0.25, 'lw': 0}
            if band_kwargs: bk.update(band_kwargs)
            valid_band = np.isfinite(pct[16]) & np.isfinite(pct[84])
            ax.fill_between(bin_centers[valid_band], pct[16][valid_band], pct[84][valid_band], **bk)

        # Median (or plot each percentile if 50 not requested)
        lk = {'color': 'k', 'lw': 2}
        if line_kwargs: lk.update(line_kwargs)
        if 50 in pct:
            v = pct[50]
            valid = np.isfinite(v)
            ax.plot(bin_centers[valid], v[valid], **lk)
        else:
            for p, v in pct.items():
                valid = np.isfinite(v)
                ax.plot(bin_centers[valid], v[valid], label=f'{p}th', **lk)

    return bin_centers, pct, counts

def plot_percentile_points(dfs, ax, *, color, label, shift=1.0,
                                    marker='o', ms=5, capsize=2, elw=1.3, **errbar_kwargs):
    """
    dfs: [p16_df, p50_df, p84_df] each with cols [x, y].
    Plots median as points with asymmetric y-errors from the 16th/84th.
    Assumes all three share the same x-grid (raises if not).
    """
    x16, y16 = dfs[0].iloc[:, 0].to_numpy(), dfs[0].iloc[:, 1].to_numpy()
    x50, y50 = dfs[1].iloc[:, 0].to_numpy(), dfs[1].iloc[:, 1].to_numpy()
    x84, y84 = dfs[2].iloc[:, 0].to_numpy(), dfs[2].iloc[:, 1].to_numpy()

    yerr_lower = np.maximum(0.0, y50 - y16)
    yerr_upper = np.maximum(0.0, y84 - y50)
    yerr = np.vstack([yerr_lower, yerr_upper])

    ax.errorbar(x50*shift, y50, yerr=yerr, fmt=marker, ms=ms,
                 color=color, ecolor=color, elinewidth=elw, capsize=capsize,
                 label=label, **errbar_kwargs)




set_latex_style()

fname = "/home/arobertson/Galacticus/paperRepositories/CalibratingGalacticus-Paper1/galacticusRuns/Low-z_High-z_Sizes/maximumPosterior/vanDerWelComparison/romanEPS_massFunction.hdf5"

bins = np.logspace(8.4, 12, 18)

#fig, axs = plt.subplots(ncols=3, figsize=(12,5))
fig, axs = plt.subplots(
    ncols=3, figsize=(12, 5), sharey=True,
    gridspec_kw={'wspace': 0.0}   # ~no gap between panels
)

for i,outputNum in enumerate([14,9,5]):

    galData = read_galacticus_output(fname, output=outputNum, bands=("u","g","z"))
    z = galData['redshift']

    blueData = [pd.read_csv(f"paperPlotData/vanDerWel_Fig8_z{z}/blue{percentile}.csv", header=None) for percentile in [16,50,84]]
    redData = [pd.read_csv(f"paperPlotData/vanDerWel_Fig8_z{z}/red{percentile}.csv", header=None) for percentile in [16,50,84]]

    Mstar = galData['Mstar']
    r50 = galData['r50_kpc']
    labels, delta = classify_starforming_quiescent(np.log10(Mstar), np.log10(galData['SFR_Msun_per_yr']), z=z, return_delta=True, delta_threshold=-1.0)

    # start plotting
    ax = axs[i]

    running_percentiles(Mstar[labels=='star-forming'], r50[labels=='star-forming'], bins, percentiles=(16, 50, 84),
                            ax=ax, plot=True, min_count=10, line_kwargs={'color':"#4477AA", 'label':'Galacticus (star forming)'}, band_kwargs={'color':'#4477AA'})

    running_percentiles(Mstar[labels=='quiescent'], r50[labels=='quiescent'], bins, percentiles=(16, 50, 84),
                            ax=ax, plot=True, min_count=10, line_kwargs={'color':'#CC6677', 'label':'Galacticus (quiescent)'}, band_kwargs={'color':'#CC6677'})


    plot_percentile_points(blueData, ax, color='#4477AA', label='3D-HST+CANDELS (star forming)', shift=0.97, ms=6, mfc='white', elw=2, capsize=3)
    plot_percentile_points(redData, ax, color='#CC6677', label='3D-HST+CANDELS (quiescent)',shift=1.03, ms=6, mfc='white', elw=2, capsize=3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(8e8,7e11)
    ax.set_ylim(0.4,22)
    if i==2:
        ax.legend(frameon=False)
    ax.set_xlabel(r"$M_\star \, / \, M_\odot$")

    ax.text(0.96, 0.04, rf"$z={z:.2f}$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=18)

for ax in axs[1:]:
    print("hello")
    #ax.tick_params(axis='y', which='both', labelleft=False, left=True, direction='in')

ax = axs[0]
ax.set_ylabel(r"$r_{50} \, / \, \mathrm{kpc}$")
ax.set_yticks([1,3,10])
ax.set_yticklabels(["1","3","10"])

plt.tight_layout()

fig.savefig('paperPlotFigures/3D-HST_CANDELS_stellarMassSizeComparison.pdf')