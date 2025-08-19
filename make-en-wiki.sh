#!/bin/bash

#SBATCH -J wiki-en-wiki
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH -c4

# Exit on error:
set -e

DIR=corpus-wiki

echo '[stanza]' >> ${DIR}/wiki-en-all.out
python tubelex.py \
	--wiki --no-categories --article-counts -n nfkc-lower \
	--lang en -x --frequencies \
	-o ${DIR}/wiki-en%.tsv.xz >> ${DIR}/wiki-en-all.out
