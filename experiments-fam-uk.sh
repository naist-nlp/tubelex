#!/bin/bash


#TODO remove from production:
# Speed things up
export HF_DATASETS_OFFLINE=1

# Exit on error:
set -e
echo subtlex-uk
python experiments/run.py --subtlex-uk --corr --fam en > experiments/fam-corr-subtlex-uk.tsv

# MLSP
for variant in '' -regex
do
	case "$variant" in
	  '')
		var_opt=''
		;;
	  -regex)
		var_opt='--tokenization regex'
		;;
	esac
	echo subtlex$variant
	python experiments/run.py $var_opt --subtlex-uk --train   --mlsp english_lcp_labels
	python experiments/run.py $var_opt --subtlex-uk --metrics --mlsp english_lcp_labels > experiments/mlsp-results-subtlex-uk${variant}.tsv
	python experiments/run.py $var_opt --subtlex-uk --corr    --mlsp english_lcp_labels > experiments/mlsp-corr-subtlex-uk${variant}.tsv
done

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
