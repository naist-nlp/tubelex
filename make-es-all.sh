#!/bin/bash

#SBATCH -J tubelex-es-all
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH -c4

# Exit on error:
set -e

DIR=frequencies

echo '[stanza]'	>> ${DIR}/tubelex-es-all.out
python tubelex.py -a -n nfkc-lower --lang es -x --frequencies -o ${DIR}/tubelex-es%.tsv.xz >> ${DIR}/tubelex-es-all.out
