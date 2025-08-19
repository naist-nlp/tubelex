#!/bin/bash

#SBATCH -J wiki-zh-wiki
#SBATCH -A lang
#SBATCH -p lang_long
#SBATCH --time=60:00:00
#SBATCH -c48

# Exit on error:
set -e

DIR=corpus-wiki

echo '[default]' >> ${DIR}/wiki-zh-all.out
python tubelex.py \
	--wiki --no-categories --article-counts -n nfkc-lower \
	--lang zh -x --frequencies \
	-o ${DIR}/wiki-zh%.tsv.xz >> ${DIR}/wiki-zh-all.out
