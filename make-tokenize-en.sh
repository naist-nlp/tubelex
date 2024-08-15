#!/bin/bash

#SBATCH -J tokenize-en
#SBATCH -A lang
#SBATCH -p lang_short
#SBATCH -c1

python tubelex.py --lang en --tokenization regex -x --tokenize
