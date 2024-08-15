#!/bin/bash

#SBATCH -J tubelex-zh
#SBATCH -A lang
#SBATCH -p lang_short
#SBATCH -c1

python tubelex.py --lang zh -x 					   -o tubelex-zh%.tsv.xz		>  tubelex-zh.out
echo '--pos' 																	>> tubelex-zh.out
python tubelex.py --lang zh -x --frequencies --pos -o tubelex-zh-pos%.tsv.xz 	>> tubelex-zh.out
