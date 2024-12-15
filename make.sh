#!/bin/bash

source config.sh	# Slack notifications - optional

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

JOB_T_EN=$(sbatch --parsable -d afterok:$JOB_EN make-tokenize-en.sh)
JOB_T_ES=$(sbatch --parsable -d afterok:$JOB_ES make-tokenize-es.sh)
JOB_T_ID=$(sbatch --parsable -d afterok:$JOB_ID make-tokenize-id.sh)
JOB_T_ZH=$(sbatch --parsable -d afterok:$JOB_ZH make-tokenize-zh.sh)
JOB_T_JA=$(sbatch --parsable -d afterok:$JOB_JA make-tokenize-ja.sh)

JOB_T_EN=$(sbatch --parsable make-tokenize-en.sh)
JOB_T_ES=$(sbatch --parsable make-tokenize-es.sh)
JOB_T_ID=$(sbatch --parsable make-tokenize-id.sh)
JOB_T_ZH=$(sbatch --parsable make-tokenize-zh.sh)
JOB_T_JA=$(sbatch --parsable make-tokenize-ja.sh)

sbatch -d afterok:$JOB_T_EN make-fasttext-en.sh 
sbatch -d afterok:$JOB_T_ES make-fasttext-es.sh 
sbatch -d afterok:$JOB_T_ID make-fasttext-id.sh 
sbatch -d afterok:$JOB_T_ZH make-fasttext-zh.sh 
sbatch -d afterok:$JOB_T_JA make-fasttext-ja.sh

for lang in zh en id ja es; do printf "${lang}\t"; head -n1 fasttext/tubelex-${lang}.vec | cut -f1 -d' '; done > fasttext/sizes.tsv

for lang in zh en id ja es; do /opt/kenlm/build/bin/lmplz -o 5 -S 80% < corpus/tokenized-${lang}.txt > kenlm/tubelex-${lang}.arpa; done &> kenlm/kenlm.out

sed -nE 's=^(Reading|[1-5]) (corpus/tokenized-)?([0-9a-z]+)(\.txt)?.*=\3=p' kenlm/kenlm.out | tr '\n' ' ' | sed -E 's/([0-9]) ([a-z]|$)/\1\n\2/g' > kenlm/sizes.tsv