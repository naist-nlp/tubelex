#!/bin/bash

# Optionally speed things up, once MLSP datasets are downloaded from HF:
export HF_DATASETS_OFFLINE=1

# Exit on error:
set -e

for task in mlsp ldt fam
	do
	if [[ "$task" = 'mlsp' ]]
	then
		langs='en es ja'
	elif [[ "$task" = 'ldt' ]]
	then
		langs='en es zh'
	else
		# task: fam
		langs='id ja zh en es'
	fi
		
	echo
	echo '==========='
	echo "TASK: $task"
	echo '==========='
	echo
	for measure in frequency range_videos range_channels range_categories \
		weighted_range gini maxmin ada juilland_d vmr gries_dp gries_dp_eq \
		rosengren_s carrol_d2 # simple_frequency, sqrt
	do
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
						cache_opt="--cache"
					fi
					mname="log_${measure}"
				fi
			else
				mname="$measure"
			fi
			echo "$mname"
			python experiments/run.py $cache_opt $transform_opt --measure $measure --tubelex $langs --corr --$task $langs > experiments/measures-${task}-corr-tubelex-${mname}.tsv
		done
	done
done