#!/bin/bash

#SBATCH -J tubelex-es1x
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -c4

echo '--lemma --pos' 																			>> tubelex-es.out
python tubelex.py --lang es -x --frequencies --form lemma --pos -o tubelex-es-lemma-pos%.tsv.xz >> tubelex-es.out
