#!/bin/bash

#SBATCH -J tubelex-en0
#SBATCH -A lang
#SBATCH -p lang_short
#SBATCH -c1

DIR=frequencies

python tubelex.py --lang en -x --clean --unique > ${DIR}/tubelex-en.out
echo '[regex]'																					    >> ${DIR}/tubelex-en.out
python tubelex.py --lang en --tokenization regex -x    --frequencies -o ${DIR}/tubelex-en-regex%.tsv.xz	>> ${DIR}/tubelex-en.out
echo '[treebank]'																				    >> ${DIR}/tubelex-en.out
python tubelex.py --lang en --tokenization treebank -x --frequencies -o ${DIR}/tubelex-en-treebank%.tsv.xz	>> ${DIR}/tubelex-en.out
