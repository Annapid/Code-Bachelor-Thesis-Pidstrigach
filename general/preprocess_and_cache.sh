#!/bin/bash -l
# Standard output and error:
#SBATCH -o /u/vstudenyak/Barry-lab-Publication_TanniDeCothiBarry2022-425bc0b/job_logs/job.out.%j
#SBATCH -e /u/vstudenyak/Barry-lab-Publication_TanniDeCothiBarry2022-425bc0b/job_logs/job.err.%j
# Initial working directory:
#SBATCH -D /u/vstudenyak/Barry-lab-Publication_TanniDeCothiBarry2022-425bc0b
# Job name
#SBATCH -J mpf_analysis
#
#SBATCH --ntasks=1
# #SBATCH --constraint="gpu"
#SBATCH --constraint="cpu"
#
# --- default case: use a single GPU on a shared node ---
# #SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40GB
#
# --- uncomment to use 2 GPUs on a shared node ---
# #SBATCH --gres=gpu:a100:2
# #SBATCH --cpus-per-task=36
# #SBATCH --mem=250000
#
# --- uncomment to use 4 GPUs on a full node ---
# #SBATCH --gres=gpu:a100:4
# #SBATCH --cpus-per-task=72
# #SBATCH --mem=500000
#
#SBATCH --mail-type=none
#SBATCH --mail-user=studenyak@cbs.mpg.de
#SBATCH --time=20:00:00

module purge
module load anaconda/3/2023.03

conda activate mpf_analysis_new

# Absolute paths on Raven. Override either of these at submission time if needed:
#   DATA_ROOT=/path/to/Paper_ExpScales_NoPreProcessing sbatch MultiplePFAnalysis/general/preprocess_and_cache.sh
REPO_ROOT="/raven/u/vstudenyak/Barry-lab-Publication_TanniDeCothiBarry2022-425bc0b"
DATA_ROOT="/raven/u/vstudenyak/Paper_ExpScales_NoPreProcessing"
RAT_IDS="${RAT_IDS:-R2470 R2474 R2478 R2481 R2482}"

cd "${REPO_ROOT}"

FAILED=0
for RAT_ID in ${RAT_IDS}; do
    echo "===== Preprocessing ${RAT_ID} ====="
    srun python "${REPO_ROOT}/MultiplePFAnalysis/general/preprocess_and_cache_all_rats.py" \
        --data-root "${DATA_ROOT}" \
        --rats "${RAT_ID}" \
        --verbose
    STATUS=$?
    if [ ${STATUS} -ne 0 ]; then
        echo "===== ${RAT_ID} failed with exit code ${STATUS} ====="
        FAILED=1
    fi
done

exit ${FAILED}
