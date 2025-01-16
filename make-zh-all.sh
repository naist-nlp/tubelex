#!/bin/bash

#SBATCH -J tubelex-zh-all
#SBATCH -A lang
#SBATCH -p lang_short
#SBATCH -c96

# Exit on error:
set -e

DIR=frequencies

echo '[zh]'																								>> ${DIR}/tubelex-zh-all.out
python tubelex.py -a -n nfkc-lower --lang zh -x --frequencies       -o ${DIR}/tubelex-zh%.tsv.xz		>> ${DIR}/tubelex-zh-all.out
