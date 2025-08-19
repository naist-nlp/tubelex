#!/bin/bash

#SBATCH -J wiki-en-wiki-1m
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH -c4

# Exit on error:
set -e

DIR=corpus-wiki

echo '[1M Stanza]' >> ${DIR}/wiki-en-1m.out
python tubelex.py \
	--wiki --no-categories --article-counts -n nfkc-lower \
	--lang en -x --frequencies \
	-R 1000000 \
	-o ${DIR}/wiki-en-1m%.tsv.xz >> ${DIR}/wiki-en-1m.out
