#!/bin/bash

source config.sh	# Slack notifications - optional

JOB_EN=$(sbatch --parsable make-en0.sh)
JOB_ID=$(sbatch --parsable make-id0.sh)
JOB_ES=$(sbatch --parsable make-es0.sh)
JOB_ZH=$(sbatch --parsable make-zh.sh)
JOB_JA=$(sbatch --parsable make-ja.sh)

sbatch -d afterok:$JOB_EN make-en1.sh
sbatch -d afterok:$JOB_EN make-en1x.sh
sbatch -d afterok:$JOB_ID make-id1.sh 
sbatch -d afterok:$JOB_ID make-id1x.sh
sbatch -d afterok:$JOB_ES make-es1.sh
sbatch -d afterok:$JOB_ES make-es1x.sh

sbatch -d afterok:$JOB_EN make-tokenize-en.sh 
sbatch -d afterok:$JOB_ES make-tokenize-es.sh 
sbatch -d afterok:$JOB_ID make-tokenize-id.sh 
sbatch -d afterok:$JOB_ZH make-tokenize-zh.sh 
sbatch -d afterok:$JOB_JA make-tokenize-ja.sh

# See README.md for the following:
#
# Build data/subimdb.tsv using files extracted from:
# https://zenodo.org/records/2552407/files/SubIMDB_All_Individual.tar?download=1
#
# python tubelex.py --lang en --frequencies --tokenized-files SubIMDB_All_Individual/subtitles -o data/subimdb.tsv
#
# Build data/laborotvspeech.tsv using laborotvspeech 1+2 (merged) from:
# https://laboro.ai/activity/column/engineer/laborotvspeech2/
#
# python tubelex.py --lang en --frequencies --laborotv --tokenized-files laborotvspeech -o data/laborotvspeech.tsv
#
# Build data/hkust_mtsc.tsv using files extracted from
# https://catalog.ldc.upenn.edu/LDC2005T32
#
# python tubelex.py --lang zh --frequencies --hkust-mtsc --tokenized-files LDC2005T32/hkust_mtsc_p1tr/data -o data/hkust-mtsc.tsv