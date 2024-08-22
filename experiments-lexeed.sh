#!/bin/bash


#TODO remove from production:
# Speed things up
export HF_DATASETS_OFFLINE=1

# Exit on error:
set -e

echo
echo '==========='
echo 'Familiarity'
echo '==========='
echo
for corpus in wordfreq tubelex tubelex-entertainment tubelex-comedy wiki os
do
	if [[ "$corpus" =~ - ]]
	then
		corpus_opt="--cat ${corpus#*-} --${corpus%-*}"
	else
		corpus_opt="--${corpus}"
	fi
	echo "$corpus"
	python experiments/run.py $corpus_opt     id ja zh                --corr --fam id ja zh > experiments/fam-corr-${corpus}.tsv
	if [[ "$corpus" =~ ^tubelex ]]
	then
		python experiments/run.py $corpus_opt id ja --form lemma 		  --corr --fam id ja    > experiments/fam-corr-${corpus}-lemma.tsv
		python experiments/run.py $corpus_opt    ja --form base 		  --corr --fam    ja    > experiments/fam-corr-${corpus}-base.tsv
		python experiments/run.py $corpus_opt id    --tokenization regex  --corr --fam id       > experiments/fam-corr-${corpus}-regex.tsv
	fi
done
echo csj-lemma
python experiments/run.py --form lemma --csj 						      --corr --fam ja 		> experiments/fam-corr-csj-lemma.tsv

# echo subtlex
# python experiments/run.py --subtlex zh                                --corr --fam zh    	> experiments/fam-corr-subtlex.tsv
# 
# rm -r experiments/models experiments/output experiments/output.tsv
# 
# echo Aggregating
# 
# python experiments/aggregate_results.py
# 
# echo Done.
