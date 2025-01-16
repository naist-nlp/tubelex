#!/bin/bash

#SBATCH -J tubelex-id-all
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH -c4

# Exit on error:
set -e

DIR=frequencies

echo '[stanza]' >> ${DIR}/tubelex-id-all.out
python tubelex.py -a -n nfkc-lower --lang id -x --frequencies -o ${DIR}/tubelex-id%.tsv.xz >> ${DIR}/tubelex-id-all.out
