#!/bin/bash

#SBATCH -J tubelex-en-all
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -c4

# Exit on error:
set -e

DIR=frequencies

echo '[stanza]'	>> ${DIR}/tubelex-en-all.out
python tubelex.py -a -n nfkc-lower --lang en -x --frequencies -o ${DIR}/tubelex-en%.tsv.xz >> ${DIR}/tubelex-en-all.out
