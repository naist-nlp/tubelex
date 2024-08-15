#!/bin/bash

#SBATCH -J tubelex-en1x
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -c4

echo '--lemma --pos' 																			>> tubelex-en.out
python tubelex.py --lang en -x --frequencies --form lemma --pos -o tubelex-en-lemma-pos%.tsv.xz >> tubelex-en.out