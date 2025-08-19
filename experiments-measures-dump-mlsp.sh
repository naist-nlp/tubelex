#!/bin/bash

# Optionally speed things up, once MLSP datasets are downloaded from HF:
export HF_DATASETS_OFFLINE=1

# Exit on error:
set -e

task=mlsp
langs='en es ja'

mkdir -p mlsp_w_tubelex

for variant in '' -lemma -base -regex
do
	for measure_method in frequency range
	do
		for measure_variant in '' '_videos' '_channels'
		do
			if [[ "$measure_method" = 'frequency' ]] && [[ -n "$measure_variant" ]]
			then
				# No variants for frequency:
				continue
			fi
			measure="${measure_method}${measure_variant}"
			mname="log_${measure}"
			langs="en es ja"
			ofiles="mlsp_w_tubelex/mlsp_w_tubelex-${mname}-en${variant}.tsv mlsp_w_tubelex/mlsp_w_tubelex-${mname}-es${variant}.tsv mlsp_w_tubelex/mlsp_w_tubelex-${mname}-ja${variant}.tsv"
			case "$variant" in
			  '')
				var_opt=''
				;;
			  -lemma)
				var_opt='--form lemma'
				;;
			  -base)
				var_opt='--form base'
				langs="ja"
				ofiles="mlsp_w_tubelex/mlsp_w_tubelex-${mname}-ja${variant}.tsv"
				;;
			  -regex)
				var_opt='--tokenization regex'
				langs="en es"
				ofiles="mlsp_w_tubelex/mlsp_w_tubelex-${mname}-en${variant}.tsv mlsp_w_tubelex/mlsp_w_tubelex-${mname}-es${variant}.tsv"
				;;
			esac
			
			python experiments/run.py --log-measure --smooth --measure $measure --tubelex $langs $var_opt --corr --$task $langs \
				-o $ofiles > /dev/null # do not need the correlation actually
		done
	done
done