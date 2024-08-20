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


echo activ-es
python experiments/run.py --activ-es --train   --mlsp spanish_lcp_labels
python experiments/run.py --activ-es --metrics --mlsp spanish_lcp_labels > experiments/mlsp-results-activ-es.tsv
python experiments/run.py --activ-es --corr    --mlsp spanish_lcp_labels > experiments/mlsp-corr-activ-es.tsv


echo
echo '==='
echo 'LDT'
echo '==='
echo
echo activ-es
python experiments/run.py --activ-es --corr --ldt es > experiments/ldt-corr-activ-es.tsv


rm -r experiments/models experiments/output experiments/output.tsv

echo Aggregating

# python experiments/aggregate_results.py

echo Done.
