#!/bin/bash

#SBATCH -J tokenize-es
#SBATCH -A lang
#SBATCH -p lang_long
#SBATCH -t 12:00:00
#SBATCH -c1

python tubelex.py --lang es --tokenization regex -x --tokenize --removed-addresses corpus/tokenized-es-removed-addresses.json

