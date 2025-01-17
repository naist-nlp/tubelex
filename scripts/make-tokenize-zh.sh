#!/bin/bash

#SBATCH -J tokenize-zh
#SBATCH -A lang
#SBATCH -p lang_short
#SBATCH -c1

python tubelex.py --lang zh  -x --tokenize --removed-addresses corpus/tokenized-zh-removed-addresses.json
