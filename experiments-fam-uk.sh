#!/bin/bash


#TODO remove from production:
# Speed things up
export HF_DATASETS_OFFLINE=1

# Exit on error:
set -e
echo subtlex-uk

# MLSP

echo
echo '==='
echo 'LDT'
echo '==='
echo
echo "subtlex-uk"
python experiments/run.py --subtlex-uk                   --corr --ldt en > experiments/ldt-corr-subtlex-uk.tsv
python experiments/run.py --subtlex-uk --tokenization regex --corr --ldt en    > experiments/ldt-corr-subtlex-uk-regex.tsv


# 
# rm -r experiments/models experiments/output experiments/output.tsv
# 
# echo Aggregating
# 
# python experiments/aggregate_results.py
# 
# echo Done.
