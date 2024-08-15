#!/bin/bash

#SBATCH -J tokenize-es
#SBATCH -A lang
#SBATCH -p lang_short
#SBATCH -c1

python tubelex.py --lang es --tokenization regex -x --tokenize
