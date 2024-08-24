#!/bin/bash


#TODO remove from production:
# Speed things up
export HF_DATASETS_OFFLINE=1

# Exit on error:
set -e

alt_opt='--glasgow --moreno-martinez'

echo
echo '==========='
echo 'Familiarity'
echo '==========='
echo
for corpus in tubelex tubelex-entertainment tubelex-comedy wordfreq wiki os
do
	cache_opt=''
	if [[ "$corpus" =~ - ]]
	then
		corpus_opt="$alt_opt --cat ${corpus#*-} --${corpus%-*}"
	else
		corpus_opt="$alt_opt --${corpus}"
		if [[ "$corpus" = 'tubelex' ]]
		then
			cache_opt="--cache"
		fi
	fi
	echo "$corpus"
	python experiments/run.py $cache_opt $corpus_opt     en es           --corr --fam en es > experiments/fam-alt-corr-${corpus}.tsv
	if [[ "$corpus" =~ ^tubelex ]]
	then
		python experiments/run.py $corpus_opt en es --form lemma         --corr --fam en es  > experiments/fam-alt-corr-${corpus}-lemma.tsv
		python experiments/run.py $corpus_opt en es --tokenization regex --corr --fam en es  > experiments/fam-alt-corr-${corpus}-regex.tsv
	fi
done
echo subtlex-uk
python experiments/run.py $alt_opt --subtlex-uk --corr --fam en > experiments/fam-alt-corr-subtlex-uk.tsv
echo subtlex
python experiments/run.py $alt_opt --subtlex en es    						      --corr --fam en es   > experiments/fam-alt-corr-subtlex.tsv
echo espal
python experiments/run.py $alt_opt --espal 										  --corr --fam es 		> experiments/fam-alt-corr-espal.tsv
echo alonso
python experiments/run.py $alt_opt --alonso 									  --corr --fam es 		> experiments/fam-alt-corr-alonso.tsv
echo activ-es
python experiments/run.py $alt_opt --activ-es 								      --corr --fam es  		> experiments/fam-alt-corr-activ-es.tsv
echo gini
python experiments/run.py $alt_opt --minus --gini en 							  --corr --fam en  	    > experiments/fam-alt-corr-gini.tsv


# 
# rm -r experiments/models experiments/output experiments/output.tsv
# 
# echo Aggregating
# 
# python experiments/aggregate_results.py
# 
# echo Done.
