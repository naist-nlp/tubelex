#!/bin/bash

#SBATCH -J wiki-es-wiki
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH -c4

# Exit on error:
set -e

DIR=corpus-wiki

echo '[stanza]' >> ${DIR}/wiki-es-all.out
python tubelex.py \
	--wiki --no-categories --article-counts -n nfkc-lower \
	--lang es -x --frequencies \
	-o ${DIR}/wiki-es%.tsv.xz >> ${DIR}/wiki-es-all.out
