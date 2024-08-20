#!/bin/bash


#TODO remove from production:
# Speed things up
export HF_DATASETS_OFFLINE=1

# Exit on error:
set -e

echo '===='
echo 'MLSP'
echo '===='
echo


echo espal
python experiments/run.py --espal --train   --mlsp spanish_lcp_labels
python experiments/run.py --espal --metrics --mlsp spanish_lcp_labels > experiments/mlsp-results-espal.tsv
python experiments/run.py --espal --corr    --mlsp spanish_lcp_labels > experiments/mlsp-corr-espal.tsv


echo
echo '==='
echo 'LDT'
echo '==='
echo
echo espal
python experiments/run.py --espal --corr --ldt es > experiments/ldt-corr-espal.tsv


rm -r experiments/models experiments/output experiments/output.tsv

echo Aggregating

# python experiments/aggregate_results.py

echo Done.
