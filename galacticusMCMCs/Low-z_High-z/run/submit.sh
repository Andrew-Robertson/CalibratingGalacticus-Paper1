#!/bin/bash
#SBATCH --time=120:00:00   # walltime
#SBATCH --ntasks=16   # number of tasks (i.e. number of Galacticus.exe that will run)
#SBATCH --cpus-per-task=1 # number of CPUs to assign to each task
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=10G   # memory per CPU core
#SBATCH -J "noSizes_broaderPriors_noLeauthaudMinMstar_varyHandCalibratedModel"   # job name
#SBATCH --mail-user=arobertson@carnegiescience.edu   # email address
#SBATCH --error=myLogFile.err # Send output to a log file
#SBATCH --output=myLogFile.out

# Notify at the beginning, end of job and on failure.
#SBATCH --mail-type=FAIL
module load conda
conda activate galacticus-dust

# Change directory to the location from which this job was submitted
cd $SLURM_SUBMIT_DIR
# Disable core-dumps (not useful unless you know what you're doing with them)
ulimit -c 0
export GFORTRAN_ERROR_DUMPCORE=NO
# Ensure there are no CPU time limits imposed.
ulimit -t unlimited
# Tell OpenMP to use all available CPUs on this node.
export OMP_NUM_THREADS=1
# Run Galacticus.
mpirun -np 16 $GALACTICUS_EXEC_PATH/Galacticus.exeMPI mcmcConfig.xml


python /home/arobertson/Galacticus/Galacticus-dust-modelling/Galacticus-dust-modelling/dust_model/scripts/postProcessMCMC.py
# Generate (and then run) a script that will produce the hdf5 output for the maximum likelihood model (and plot the results)
python /home/arobertson/Galacticus/Galacticus-dust-modelling/Galacticus-dust-modelling/dust_model/scripts/generateSingleLikelihoodSlurmScript.py
sbatch singleLikelihood.sh
# Generate and then run a script to run a mass function of halos and plot the outputAnalysis
python /home/arobertson/Galacticus/Galacticus-dust-modelling/Galacticus-dust-modelling/dust_model/scripts/generateMassFunctionSlurmScript.py --model-change /home/arobertson/Galacticus/forked_galacticus/galacticus/parameters/Roman/hierarchicalParameters/fineTuningFiles/additionalQuantities.xml
# Generate and then run a script to run a mass function of halos and plot the outputAnalysis (maximumPosterior)
python /home/arobertson/Galacticus/Galacticus-dust-modelling/Galacticus-dust-modelling/dust_model/scripts/generateMassFunctionSlurmScript.py --parameterChoice maximumPosterior --model-change /home/arobertson/Galacticus/forked_galacticus/galacticus/parameters/Roman/hierarchicalParameters/fineTuningFiles/additionalQuantities.xml

