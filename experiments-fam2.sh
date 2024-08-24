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
# for corpus in wordfreq tubelex tubelex-entertainment tubelex-comedy wiki os
# do
# 	if [[ "$corpus" =~ - ]]
# 	then
# 		corpus_opt="--cat ${corpus#*-} --${corpus%-*}"
# 	else
# 		corpus_opt="--${corpus}"
# 	fi
# 	echo "$corpus"
# 	python experiments/run.py $corpus_opt     id ja zh en es              --corr --fam id ja zh en es > experiments/fam-corr-${corpus}.tsv
# 	if [[ "$corpus" =~ ^tubelex ]]
# 	then
# 		python experiments/run.py $corpus_opt id ja en es --form lemma         --corr --fam id ja en es  > experiments/fam-corr-${corpus}-lemma.tsv
# 		python experiments/run.py $corpus_opt    ja       --form base 		   --corr --fam    ja        > experiments/fam-corr-${corpus}-base.tsv
# 		python experiments/run.py $corpus_opt id    en es --tokenization regex --corr --fam id    en es  > experiments/fam-corr-${corpus}-regex.tsv
# 	fi
# done
echo subtlex-uk
python experiments/run.py --subtlex-uk --corr --fam en > experiments/fam-corr-subtlex-uk.tsv
echo csj-lemma
python experiments/run.py --form lemma --csj 						      --corr --fam ja 		> experiments/fam-corr-csj-lemma.tsv
echo subtlex
python experiments/run.py --subtlex zh en es    						  --corr --fam zh en es > experiments/fam-corr-subtlex.tsv
echo espal
python experiments/run.py --espal 										  --corr --fam es 		> experiments/fam-corr-espal.tsv
echo alonso
python experiments/run.py --alonso 										  --corr --fam es 		> experiments/fam-corr-alonso.tsv
echo activ-es
python experiments/run.py --activ-es 								      --corr --fam es  		> experiments/fam-corr-activ-es.tsv
echo gini
python experiments/run.py --minus --gini en ja 							  --corr --fam en ja  	> experiments/fam-corr-gini.tsv
echo subimdb
python experiments/run.py --subimdb 									  --corr  --fam en   	> experiments/fam-corr-subimdb.tsv
echo laborotv
python experiments/run.py -D ipadic --laborotv 						      --corr --fam ja 		> experiments/fam-corr-laborotv.tsv


# 
# rm -r experiments/models experiments/output experiments/output.tsv
# 
# echo Aggregating
# 
# python experiments/aggregate_results.py
# 
# echo Done.
