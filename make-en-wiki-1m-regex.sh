#!/bin/bash

#SBATCH -J wiki-en-wiki-1m-regex
#SBATCH -A lang
#SBATCH -p lang_long
#SBATCH --time=100:00:00
#SBATCH -c48

# Exit on error:
set -e

DIR=corpus-wiki

echo '[1M-regex]' >> ${DIR}/wiki-en-1m-regex.out
python tubelex.py \
	--wiki --no-categories --article-counts -n nfkc-lower \
	--lang en -x --frequencies \
	--tokenization regex -R 1000000 \
	-o ${DIR}/wiki-en-1m-regex%.tsv.xz >> ${DIR}/wiki-en-1m-regex.out
