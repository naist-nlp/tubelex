#!/bin/bash

source config.sh	# Slack notifications - optional

JOB_EN=$(sbatch --parsable make-en0.sh)
JOB_ID=$(sbatch --parsable make-id0.sh)
JOB_ES=$(sbatch --parsable make-es0.sh)
sbatch make-zh.sh 
sbatch make-ja.sh

sbatch -d afterok:$JOB_EN make-en1.sh
sbatch -d afterok:$JOB_EN make-en1x.sh
sbatch -d afterok:$JOB_ID make-id1.sh 
sbatch -d afterok:$JOB_ID make-id1x.sh
sbatch -d afterok:$JOB_ES make-es1.sh
sbatch -d afterok:$JOB_ES make-es1x.sh

sbatch make-tokenize-en.sh 
sbatch make-tokenize-es.sh 
sbatch make-tokenize-id.sh 
sbatch make-tokenize-zh.sh 
sbatch make-tokenize-ja.sh

# Build data/subimdb.tsv using files extracted from
# https://zenodo.org/records/2552407/files/SubIMDB_All_Individual.tar?download=1
#
# python tubelex.py --lang en --frequencies --tokenized-files SubIMDB_All_Individual/subtitles -o data/subimdb.tsv


# laborotvspeech 1+2 (merged)
# https://laboro.ai/activity/column/engineer/laborotvspeech2/
# python tubelex.py --lang en -x --frequencies --laborotv --tokenized-files laborotvspeech -o data/laborotvspeech.tsv.xz