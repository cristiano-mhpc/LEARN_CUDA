#!/bin/bash -l
#SBATCH -A tra24_ictp_np
#SBATCH -p boost_usr_prod
#SBATCH --time 4:15:00       # format: HH:MM:SS
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=10
#SBATCH --job-name=myjob
#SBATCH --output=output.txt
#SBATCH --error=myjob.err
#SBATCH --mail-user=cdtica1@up.edu.ph


module load boost/1.83.0--openmpi--4.1.6--gcc--12.2.0

module load cuda

./trapezoid 10000 2 50 -3 3





