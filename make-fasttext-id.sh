#!/bin/bash

#SBATCH -J fasttext-id
#SBATCH -A lang
#SBATCH -p lang_long
#SBATCH -t 100:00:00
#SBATCH -c 96

mkdir -p fasttext
fasttext skipgram -input corpus/tokenized-id.txt -output fasttext/tubelex-id -thread 96
