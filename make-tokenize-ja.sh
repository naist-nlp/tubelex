#!/bin/bash

#SBATCH -J tokenize-ja
#SBATCH -A lang
#SBATCH -p lang_short
#SBATCH -c1

python tubelex.py --lang ja  -x --tokenize
