# CalibratingGalacticus-Paper1
This repositry contains parameter files, catalogues, plotting scripts, etc. related to the paper "Accelerated calibration of semi-analytic galaxy formation models"

## Getting the data

Many of the figures require Galactcicus catalogues (~4 GB) archived on Zenodo:

**DOI:** [10.5281/zenodo.16952803](https://zenodo.org/records/16952804)

To download and unpack them into the expected location, from the **root of this repository**, run:

```bash
curl -L "https://zenodo.org/records/16952804/files/galacticusRuns.zip?download=1" -o galacticusRuns.zip
unzip -n galacticusRuns.zip
```

## Plotting scripts

The `paper_plots` directory contains scripts that generate the figures from "Accelerated calibration of semi-analytic galaxy formation models". For example, Figure 1 from the paper can be generated with `python cornerPlot_lowzSHMR.py` run from the `paper_plots` directory. All the files necessary to generate the plots should be available as part of this repository (if you have followed the "Getting the data" instructions above). Various publicly available packages may need to be installed, but hopefully everything is easily installable (please reach out to Andrew Robertson if anything doesn't work, and apologies that this repositry was an afterthought!).

## MCMC runs

The `galacticusMCMCs` directory contains files relevant to the MCMC runs presented in the paper. Taking the `galacticusMCMCs/Low-z_High-z_Sizes` directory as an example (this is "full calibration" in Fig. 3), the sub-directories are:
- `mcmcParameterChanges`: Galacticus [parameter changes files](https://github.com/galacticusorg/galacticus/wiki/Tutorial:-Introduction-to-Galacticus-parameter-files#changing-parameters) for the maximum-likelihood, maximum-posterior, etc. model
- `mcmcPlots`: corner plot, trace plot, etc. of the MCMC
- `run`: the Galacticus config file and submission script used to run the MCMC. Note that the submission script is for the Carnegie Science OBS HPC and will not just generically work. Also, I have tried to update the config files (`mcmcConfig.xml`) to reference files within the `galacticusParameters` directory, but this is not the directory structure within which the MCMCs were originally run, so please let me know if you notice any errors!
- `singleLikelihood`: a Galacticus run with the maximum likelihood parameters from the MCMC chain of a single likelihood evaluation. These are used (for example) in Fig. 5.


