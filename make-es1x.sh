#!/bin/bash

#SBATCH -J tubelex-es1x
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -c4

DIR=frequencies

echo '--lemma --pos' 																			>> ${DIR}/tubelex-es.out
python tubelex.py --lang es -x --frequencies --form lemma --pos -o ${DIR}/tubelex-es-lemma-pos%.tsv.xz >> ${DIR}/tubelex-es.out
