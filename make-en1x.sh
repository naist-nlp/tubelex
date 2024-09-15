#!/bin/bash

#SBATCH -J tubelex-en1x
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -c4

DIR=frequencies

echo '--lemma --pos' 																			>> ${DIR}/tubelex-en.out
python tubelex.py --lang en -x --frequencies --form lemma --pos -o ${DIR}/tubelex-en-lemma-pos%.tsv.xz >> ${DIR}/tubelex-en.out