#!/bin/bash

#SBATCH -J tubelex-zh
#SBATCH -A lang
#SBATCH -p lang_short
#SBATCH -c1

python tubelex.py --lang zh -x 					   -o ${DIR}/tubelex-zh%.tsv.xz		>  tubelex-zh.out
echo '--pos' 																	>> ${DIR}/tubelex-zh.out
python tubelex.py --lang zh -x --frequencies --pos -o ${DIR}/tubelex-zh-pos%.tsv.xz 	>> ${DIR}/tubelex-zh.out
