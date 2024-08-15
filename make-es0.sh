#!/bin/bash

#SBATCH -J tubelex-es0
#SBATCH -A lang
#SBATCH -p lang_short
#SBATCH -c1

python tubelex.py --lang es -x --clean --unique > tubelex-es.out
echo '[regex]'																					    >> tubelex-es.out
python tubelex.py --lang es --tokenization regex -x    --frequencies -o tubelex-es-regex%.tsv.xz	>> tubelex-es.out
