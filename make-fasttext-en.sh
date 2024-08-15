#!/bin/bash

#SBATCH -J fasttext-en
#SBATCH -A lang
#SBATCH -p lang_long
#SBATCH -t 100:00:00
#SBATCH -c 96

mkdir -p fasttext
fasttext skipgram -input corpus/tokenized-en.txt -output fasttext/tubelex-en -thread 96
