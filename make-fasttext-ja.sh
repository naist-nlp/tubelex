#!/bin/bash

#SBATCH -J fasttext-ja
#SBATCH -A lang
#SBATCH -p lang_long
#SBATCH -t 100:00:00
#SBATCH -c 96

mkdir -p fasttext
fasttext skipgram -input corpus/tokenized-ja.txt -output fasttext/tubelex-ja -thread 96
