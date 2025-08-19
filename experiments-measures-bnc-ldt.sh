#!/bin/bash

# Optionally speed things up, once MLSP datasets are downloaded from HF:
export HF_DATASETS_OFFLINE=1

# Exit on error:
set -e

for task in ldt # mlsp ldt fam
do
	langs='en'
		
	echo
	echo '==========='
	echo "TASK: $task"
	echo '==========='
	echo
	
	for measure_method in frequency range lyne_d3 rosengren_s s2 juilland_d \
		gries_dp carrol_d2 sort_gini
	do
		for measure_variant in '' '_videos'
		do
			if [[ "$measure_method" = 'frequency' ]] && [[ -n "$measure_variant" ]]
			then
				# No variants for frequency:
				continue
			fi
			measure="${measure_method}${measure_variant}"
			for transform_opt in '--log-measure --smooth' '' # '--sqrt-measure --zero-clip'
			do
				cache_opt=''
				if [[ -n "$transform_opt" ]]
				then
					if [[ "$transform_opt" = '--sqrt-measure --zero-clip' ]]
					then
						mname="sqrt_${measure}"
					else
						if [[ "$measure" = 'frequency' ]]
						then
							# Cache LOG frequency
							cache_opt='--cache'
						fi
						mname="log_${measure}"
					fi
				else
					mname="${measure}"
				fi
				echo "$mname"
				python experiments/run.py --cached bnc --tokenization regex $cache_opt $transform_opt --measure $measure --bnc --corr --$task $langs > experiments/measures-${task}-corr-bnc-${mname}.tsv
			done
		done
	done
done