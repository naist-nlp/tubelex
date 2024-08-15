#!/bin/bash

#SBATCH -J tubelex-en1
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -c4

echo '[stanza]'	>> tubelex-en.out
python tubelex.py --lang en -x --frequencies -o tubelex-en%.tsv.xz >> tubelex-en.out
