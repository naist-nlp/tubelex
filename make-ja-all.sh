#!/bin/bash

#SBATCH -J tubelex-ja-all
#SBATCH -A lang
#SBATCH -p lang_short
#SBATCH -c1

# Exit on error:
set -e

DIR=frequencies

echo '[unidic-lite]'														>> ${DIR}/tubelex-ja-all.out
python tubelex.py -a -n nfkc-lower --lang ja -x --frequencies -o ${DIR}/tubelex-ja%.tsv.xz	>> ${DIR}/tubelex-ja-all.out
