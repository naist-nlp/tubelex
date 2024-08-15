#!/bin/bash

#SBATCH -J tubelex-es1
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -c4

echo '[stanza]'	>> tubelex-es.out
python tubelex.py --lang es -x --frequencies -o tubelex-es%.tsv.xz >> tubelex-es.out
