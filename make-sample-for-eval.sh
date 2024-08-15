#!/bin/bash

#SBATCH -J tubelex-sample-for-eval
#SBATCH -A lang
#SBATCH -p lang_short
#SBATCH -c1

for lang in en es id zh ja
do
	python sample_for_eval.py --lang $lang -x
done