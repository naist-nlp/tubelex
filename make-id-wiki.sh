#!/bin/bash

#SBATCH -J wiki-id-wiki
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH -c4

# Exit on error:
set -e

DIR=corpus-wiki

echo '[stanza]' >> ${DIR}/wiki-id-all.out
python tubelex.py \
	--wiki --no-categories --article-counts -n nfkc-lower \
	--lang id -x --frequencies \
	-o ${DIR}/wiki-id%.tsv.xz >> ${DIR}/wiki-id-all.out
