#!/bin/bash

#SBATCH -J tubelex-id0
#SBATCH -A lang
#SBATCH -p lang_short
#SBATCH -c1

DIR=frequencies

python tubelex.py --lang id -x --clean --unique > ${DIR}/tubelex-id.out
echo '[regex]'																					    >> ${DIR}/tubelex-id.out
python tubelex.py --lang id --tokenization regex -x    --frequencies -o ${DIR}/tubelex-id-regex%.tsv.xz	>> ${DIR}/tubelex-id.out
