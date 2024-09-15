#!/bin/bash

#SBATCH -J tubelex-es1
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -c4

DIR=frequencies

echo '[stanza]'	>> ${DIR}/tubelex-es.out
python tubelex.py --lang es -x --frequencies -o ${DIR}/tubelex-es%.tsv.xz >> ${DIR}/tubelex-es.out
