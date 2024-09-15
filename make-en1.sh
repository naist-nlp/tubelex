#!/bin/bash

#SBATCH -J tubelex-en1
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -c4

DIR=frequencies

echo '[stanza]'	>> ${DIR}/tubelex-en.out
python tubelex.py --lang en -x --frequencies -o ${DIR}/tubelex-en%.tsv.xz >> ${DIR}/tubelex-en.out
