#!/bin/bash

#SBATCH -J tokenize-id
#SBATCH -A lang
#SBATCH -p lang_short
#SBATCH -c1

python tubelex.py --lang id --tokenization regex -x --tokenize
