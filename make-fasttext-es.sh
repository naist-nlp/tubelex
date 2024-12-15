#!/bin/bash

#SBATCH -J fasttext-es
#SBATCH -A lang
#SBATCH -p lang_short
#SBATCH -c 96

source make-fasttext-config.sh "$1"
fasttext $params -input corpus/tokenized-es.txt -output "fasttext/tubelex-es$variant" -thread 96
