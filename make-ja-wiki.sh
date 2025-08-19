#!/bin/bash

#SBATCH -J wiki-ja-wiki
#SBATCH -A lang
#SBATCH -p lang_long
#SBATCH --time=60:00:00
#SBATCH -c48

# Exit on error:
set -e

DIR=corpus-wiki

echo '[default]' >> ${DIR}/wiki-ja-all.out
python tubelex.py \
	--wiki --no-categories --article-counts -n nfkc-lower \
	--lang ja -x --frequencies \
	-o ${DIR}/wiki-ja%.tsv.xz >> ${DIR}/wiki-ja-all.out
