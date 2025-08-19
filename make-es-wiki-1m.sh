#!/bin/bash

#SBATCH -J wiki-es-wiki-1m
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH -c4

# Exit on error:
set -e

DIR=corpus-wiki

echo '[1M Stanza]' >> ${DIR}/wiki-es-1m.out
python tubelex.py \
	--wiki --no-categories --article-counts -n nfkc-lower \
	--lang es -x --frequencies \
	-R 1000000 \
	-o ${DIR}/wiki-es-1m%.tsv.xz >> ${DIR}/wiki-es-1m.out
