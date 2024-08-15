#!/bin/bash

#SBATCH -J fasttext-zh
#SBATCH -A lang
#SBATCH -p lang_long
#SBATCH -t 100:00:00
#SBATCH -c 96

mkdir -p fasttext
fasttext skipgram -input corpus/tokenized-zh.txt -output fasttext/tubelex-zh -thread 96
