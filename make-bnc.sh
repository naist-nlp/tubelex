#!/bin/bash

#SBATCH -J bnc
#SBATCH -A lang
#SBATCH -p lang_short
#SBATCH -c2

# Exit on error:
set -e

DIR=frequencies

python tubelex.py \                      
	--bnc --text-counts -n nfkc-lower \
	--tokenized data/bnc \
	--lang en -x --frequencies \
	-o ${DIR}/bnc%.tsv.xz >> ${DIR}/bnc.out