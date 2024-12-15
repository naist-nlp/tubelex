#!/bin/bash

#SBATCH -J tokenize-en
#SBATCH -A lang
#SBATCH -p lang_long
#SBATCH -t 12:00:00
#SBATCH -c1

python tubelex.py --lang en --tokenization regex -x --tokenize --removed-addresses corpus/tokenized-en-removed-addresses.json
