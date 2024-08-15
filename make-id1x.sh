#!/bin/bash

#SBATCH -J tubelex-id1x
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -c4

echo '--lemma --pos' 																			>> tubelex-id.out
python tubelex.py --lang id -x --frequencies --form lemma --pos -o tubelex-id-lemma-pos%.tsv.xz >> tubelex-id.out