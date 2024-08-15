#!/bin/bash

#SBATCH -J tubelex-en0
#SBATCH -A lang
#SBATCH -p lang_short
#SBATCH -c1

python tubelex.py --lang en -x --clean --unique > tubelex-en.out
echo '[regex]'																					    >> tubelex-en.out
python tubelex.py --lang en --tokenization regex -x    --frequencies -o tubelex-en-regex%.tsv.xz	>> tubelex-en.out
echo '[treebank]'																				    >> tubelex-en.out
python tubelex.py --lang en --tokenization treebank -x --frequencies -o tubelex-en-treebank%.tsv.xz	>> tubelex-en.out
