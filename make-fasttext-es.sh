#!/bin/bash

#SBATCH -J fasttext-es
#SBATCH -A lang
#SBATCH -p lang_long
#SBATCH -t 100:00:00
#SBATCH -c 96

mkdir -p fasttext
fasttext skipgram -input corpus/tokenized-es.txt -output fasttext/tubelex-es -thread 96
