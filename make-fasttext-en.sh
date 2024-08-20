#!/bin/bash

#SBATCH -J fasttext-en
#SBATCH -A lang
#SBATCH -p lang_long
#SBATCH -t 100:00:00
#SBATCH -c 96

source make-fasttext-config.sh "$1"
fasttext $params -input corpus/tokenized-en.txt -output "fasttext/tubelex-en-$variant" -thread 96
