#!/bin/bash

#SBATCH -J tubelex-id0
#SBATCH -A lang
#SBATCH -p lang_short
#SBATCH -c1

python tubelex.py --lang id -x --clean --unique > tubelex-id.out
echo '[regex]'																					    >> tubelex-id.out
python tubelex.py --lang id --tokenization regex -x    --frequencies -o tubelex-id-regex%.tsv.xz	>> tubelex-id.out
