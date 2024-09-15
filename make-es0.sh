#!/bin/bash

#SBATCH -J tubelex-es0
#SBATCH -A lang
#SBATCH -p lang_short
#SBATCH -c1

DIR=frequencies

python tubelex.py --lang es -x --clean --unique > ${DIR}/tubelex-es.out
echo '[regex]'																					    >> ${DIR}/tubelex-es.out
python tubelex.py --lang es --tokenization regex -x    --frequencies -o ${DIR}/tubelex-es-regex%.tsv.xz	>> ${DIR}/tubelex-es.out
