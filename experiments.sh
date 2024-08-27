#!/bin/bash

# Optionally speed things up, once MLSP datasets are downloaded from HF:
# export HF_DATASETS_OFFLINE=1

# Exit on error:
set -e

echo '===='
echo 'MLSP'
echo '===='
echo

all="spanish_lcp_labels english_lcp_labels japanese_lcp_labels"
esen="spanish_lcp_labels english_lcp_labels"


for variant in '' -lemma -base -regex
do
	langs="es en ja"
	mlsp="$all"	
	cache_opt=''
	case "$variant" in
	  '')
		cache_opt='--cache'
		var_opt=''
		;;
	  -lemma)
		var_opt='--form lemma'
		;;
	  -base)
		var_opt='--form base'
		langs="ja"
		mlsp="japanese_lcp_labels"
		;;
	  -regex)
		var_opt='--tokenization regex'
		langs="es en"
		mlsp="$esen"
		;;
	esac
	echo tubelex$variant
	python experiments/run.py $var_opt --tubelex $langs --train   --mlsp $mlsp
	python experiments/run.py $var_opt --tubelex $langs --metrics --mlsp $mlsp > experiments/mlsp-results-tubelex${variant}.tsv
	python experiments/run.py $cache_opt $var_opt --tubelex $langs --corr    --mlsp $mlsp > experiments/mlsp-corr-tubelex${variant}.tsv
	
	echo tubelex-entertainment$variant
	python experiments/run.py $var_opt --tubelex $langs --cat entertainment --train --mlsp $mlsp
	python experiments/run.py $var_opt --tubelex $langs --cat entertainment --metrics --mlsp $mlsp > experiments/mlsp-results-tubelex-entertainment${variant}.tsv
	python experiments/run.py $var_opt --tubelex $langs --cat entertainment --corr --mlsp $mlsp    > experiments/mlsp-corr-tubelex-entertainment${variant}.tsv
	
	echo tubelex-comedy$variant
	python experiments/run.py $var_opt --tubelex $langs --cat comedy --train --mlsp $mlsp
	python experiments/run.py $var_opt --tubelex $langs --cat comedy --metrics --mlsp $mlsp > experiments/mlsp-results-tubelex-comedy${variant}.tsv
	python experiments/run.py $var_opt --tubelex $langs --cat comedy --corr    --mlsp $mlsp > experiments/mlsp-corr-tubelex-comedy${variant}.tsv
done

echo csj-lemma
python experiments/run.py --form lemma --csj --train   --mlsp japanese_lcp_labels
python experiments/run.py --form lemma --csj --metrics --mlsp japanese_lcp_labels > experiments/mlsp-results-csj-lemma.tsv
python experiments/run.py --form lemma --csj --corr    --mlsp japanese_lcp_labels > experiments/mlsp-corr-csj-lemma.tsv

echo os
python experiments/run.py --os es en ja --train   --mlsp $all
python experiments/run.py --os es en ja --metrics --mlsp $all > experiments/mlsp-results-os.tsv
python experiments/run.py --os es en ja --corr    --mlsp $all > experiments/mlsp-corr-os.tsv

echo espal
python experiments/run.py --espal --train   --mlsp spanish_lcp_labels
python experiments/run.py --espal --metrics --mlsp spanish_lcp_labels > experiments/mlsp-results-espal.tsv
python experiments/run.py --espal --corr    --mlsp spanish_lcp_labels > experiments/mlsp-corr-espal.tsv

echo alonso
python experiments/run.py --alonso --train   --mlsp spanish_lcp_labels
python experiments/run.py --alonso --metrics --mlsp spanish_lcp_labels > experiments/mlsp-results-alonso.tsv
python experiments/run.py --alonso --corr    --mlsp spanish_lcp_labels > experiments/mlsp-corr-alonso.tsv

echo activ-es
python experiments/run.py --activ-es --train   --mlsp spanish_lcp_labels
python experiments/run.py --activ-es --metrics --mlsp spanish_lcp_labels > experiments/mlsp-results-activ-es.tsv
python experiments/run.py --activ-es --corr    --mlsp spanish_lcp_labels > experiments/mlsp-corr-activ-es.tsv

echo wordfreq
python experiments/run.py --wordfreq es en ja --train   --mlsp $all
python experiments/run.py --wordfreq es en ja --metrics --mlsp $all > experiments/mlsp-results-wordfreq.tsv
python experiments/run.py --wordfreq es en ja --corr    --mlsp $all > experiments/mlsp-corr-wordfreq.tsv

echo wiki
python experiments/run.py --wiki es en ja --train   --mlsp $all
python experiments/run.py --wiki es en ja --metrics --mlsp $all > experiments/mlsp-results-wiki.tsv
python experiments/run.py --wiki es en ja --corr    --mlsp $all > experiments/mlsp-corr-wiki.tsv
	
echo gini
python experiments/run.py --minus --gini en ja --train   --mlsp english_lcp_labels japanese_lcp_labels
python experiments/run.py --minus --gini en ja --metrics --mlsp english_lcp_labels japanese_lcp_labels > experiments/mlsp-results-gini.tsv
python experiments/run.py --minus --gini en ja --corr    --mlsp english_lcp_labels japanese_lcp_labels > experiments/mlsp-corr-gini.tsv

echo laborotv
python experiments/run.py -D ipadic --laborotv --train   --mlsp japanese_lcp_labels
python experiments/run.py -D ipadic --laborotv --metrics --mlsp japanese_lcp_labels > experiments/mlsp-results-laborotv.tsv
python experiments/run.py -D ipadic --laborotv --corr    --mlsp japanese_lcp_labels > experiments/mlsp-corr-laborotv.tsv

# Doesn't improve:
# echo GINI-log
# python experiments/run.py --minus-log --gini en ja --train  \
# 	english_lcp_labels japanese_lcp_labels
# python experiments/run.py --minus-log --gini en ja --metrics \
# 	english_lcp_labels japanese_lcp_labels > experiments/mlsp-results-gini-log.tsv
# python experiments/run.py --minus-log --gini en ja --corr \
# 	english_lcp_labels japanese_lcp_labels > experiments/mlsp-corr-gini-log.tsv


echo subtlex
python experiments/run.py --subtlex es en --train   --mlsp $esen
python experiments/run.py --subtlex es en --metrics --mlsp $esen > experiments/mlsp-results-subtlex.tsv
python experiments/run.py --subtlex es en --corr    --mlsp $esen > experiments/mlsp-corr-subtlex.tsv

echo subtlex-uk
python experiments/run.py --subtlex-uk --train   --mlsp english_lcp_labels
python experiments/run.py --subtlex-uk --metrics --mlsp english_lcp_labels > experiments/mlsp-results-subtlex-uk.tsv
python experiments/run.py --subtlex-uk --corr    --mlsp english_lcp_labels > experiments/mlsp-corr-subtlex-uk.tsv

echo subimdb
python experiments/run.py --subimdb --train   --mlsp english_lcp_labels
python experiments/run.py --subimdb --metrics --mlsp english_lcp_labels   > experiments/mlsp-results-subimdb.tsv
python experiments/run.py --subimdb --corr    --mlsp english_lcp_labels   > experiments/mlsp-corr-subimdb.tsv

echo spoken-bnc
python experiments/run.py --spoken-bnc --train   --mlsp english_lcp_labels
python experiments/run.py --spoken-bnc --metrics --mlsp english_lcp_labels   > experiments/mlsp-results-spoken-bnc.tsv
python experiments/run.py --spoken-bnc --corr    --mlsp english_lcp_labels   > experiments/mlsp-corr-spoken-bnc.tsv


echo
echo '==='
echo 'LDT'
echo '==='
echo
for corpus in tubelex tubelex-entertainment tubelex-comedy wordfreq  wiki subtlex os
do
	cache_opt=''
	if [[ "$corpus" =~ - ]]
	then
		corpus_opt="--cat ${corpus#*-} --${corpus%-*}"
	else
		corpus_opt="--${corpus}"
		if [[ "$corpus" = 'tubelex' ]]
		then
			cache_opt="--cache"
		fi
	fi
	echo "$corpus"
	python experiments/run.py $cache_opt $corpus_opt     en es zh		 --corr --ldt en es zh > experiments/ldt-corr-${corpus}.tsv
	if [[ "$corpus" =~ ^tubelex ]]
	then
		python experiments/run.py $corpus_opt en es --form lemma         --corr --ldt en es    > experiments/ldt-corr-${corpus}-lemma.tsv
		python experiments/run.py $corpus_opt en es --tokenization regex --corr --ldt en es    > experiments/ldt-corr-${corpus}-regex.tsv
	fi
done
echo "gini"
python experiments/run.py --minus --gini en                              --corr --ldt en       > experiments/ldt-corr-gini.tsv
echo "subimdb"
python experiments/run.py --subimdb 				                     --corr --ldt en       > experiments/ldt-corr-subimdb.tsv
echo "spoken-bnc"
python experiments/run.py --spoken-bnc 				                     --corr --ldt en       > experiments/ldt-corr-spoken-bnc.tsv
echo "espal"
python experiments/run.py --espal 										 --corr --ldt es	   > experiments/ldt-corr-espal.tsv
echo "alonso"
python experiments/run.py --alonso 										 --corr --ldt es	   > experiments/ldt-corr-alonso.tsv
echo "activ-es"
python experiments/run.py --activ-es 									 --corr --ldt es 	   > experiments/ldt-corr-activ-es.tsv
echo "subtlex-uk"
python experiments/run.py --subtlex-uk                   --corr --ldt en > experiments/ldt-corr-subtlex-uk.tsv
echo "hkust-mtsc"
python experiments/run.py --hkust-mtsc                   --corr --ldt zh > experiments/ldt-corr-hkust-mtsc.tsv


# We don't have z-scores ready for Spanish, and means are good enough to compare various corpora, although noisier:
# echo
# echo '==========='
# echo 'LDT z score'
# echo '==========='
# echo
# for corpus in tubelex subtlex os
# do
# 	corpus_opt="--${corpus}"
# 	echo "$corpus"
# 	python experiments/run.py $corpus_opt     en zh                   --corr -z --ldt en zh 	  > experiments/ldtz-corr-${corpus}.tsv
# done
# echo "subimdb"
# python experiments/run.py --subimdb 				                     --corr -z --ldt en       > experiments/ldtz-corr-subimdb.tsv


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
		corpus_opt="--cat ${corpus#*-} --${corpus%-*}"
	else
		corpus_opt="--${corpus}"
		if [[ "$corpus" = 'tubelex' ]]
		then
			cache_opt="--cache"
		fi
	fi
	echo "$corpus"
	python experiments/run.py $cache_opt $corpus_opt id ja zh en es            --corr --fam id ja zh en es > experiments/fam-corr-${corpus}.tsv
	if [[ "$corpus" =~ ^tubelex ]]
	then
		python experiments/run.py $corpus_opt id ja en es --form lemma         --corr --fam id ja en es  > experiments/fam-corr-${corpus}-lemma.tsv
		python experiments/run.py $corpus_opt    ja       --form base 		   --corr --fam    ja        > experiments/fam-corr-${corpus}-base.tsv
		python experiments/run.py $corpus_opt id    en es --tokenization regex --corr --fam id    en es  > experiments/fam-corr-${corpus}-regex.tsv
	fi
done
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
echo spoken-bnc
python experiments/run.py --spoken-bnc 									  --corr  --fam en   	> experiments/fam-corr-spoken-bnc.tsv
echo hkust-mtsc
python experiments/run.py --hkust-mtsc 									  --corr  --fam zh   	> experiments/fam-corr-hkust-mtsc.tsv
echo laborotv
python experiments/run.py -D ipadic --laborotv 						      --corr --fam ja 		> experiments/fam-corr-laborotv.tsv




echo
echo '=============================='
echo 'Familiarity (Alternative Data)'
echo '=============================='
echo
alt_opt='--glasgow --moreno-martinez --amano'
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
	python experiments/run.py $cache_opt $corpus_opt     en es ja           --corr --fam en es ja > experiments/fam-alt-corr-${corpus}.tsv
	if [[ "$corpus" =~ ^tubelex ]]
	then
		python experiments/run.py $corpus_opt en es ja --form lemma         --corr --fam en es ja  > experiments/fam-alt-corr-${corpus}-lemma.tsv
		python experiments/run.py $corpus_opt    ja       --form base 		   --corr --fam    ja        > experiments/fam-alt-corr-${corpus}-base.tsv
		python experiments/run.py $corpus_opt en es --tokenization regex --corr --fam en es  > experiments/fam-alt-corr-${corpus}-regex.tsv
	fi
done
echo csj-lemma
python experiments/run.py $alt_opt --form lemma --csj 						      --corr --fam ja 		> experiments/fam-alt-corr-csj-lemma.tsv
echo subtlex-uk
python experiments/run.py $alt_opt --subtlex-uk --corr --fam en > experiments/fam-alt-corr-subtlex-uk.tsv
echo subtlex
python experiments/run.py $alt_opt --subtlex en es    						      --corr --fam en es    > experiments/fam-alt-corr-subtlex.tsv
echo espal
python experiments/run.py $alt_opt --espal 										  --corr --fam es 		> experiments/fam-alt-corr-espal.tsv
echo alonso
python experiments/run.py $alt_opt --alonso 									  --corr --fam es 		> experiments/fam-alt-corr-alonso.tsv
echo activ-es
python experiments/run.py $alt_opt --activ-es 								      --corr --fam es  		> experiments/fam-alt-corr-activ-es.tsv
echo gini
python experiments/run.py $alt_opt --minus --gini en ja 					      --corr --fam en ja    > experiments/fam-alt-corr-gini.tsv
echo subimdb
python experiments/run.py $alt_opt --subimdb 									  --corr  --fam en   	> experiments/fam-alt-corr-subimdb.tsv
echo spoken-bnc
python experiments/run.py $alt_opt --spoken-bnc 								  --corr  --fam en   	> experiments/fam-alt-corr-spoken-bnc.tsv
echo laborotv
python experiments/run.py $alt_opt -D ipadic --laborotv 						  --corr --fam ja 		> experiments/fam-alt-corr-laborotv.tsv

# Optionally clean up:
# rm -r experiments/models experiments/output experiments/output.tsv experiments/cache

echo Aggregating

python experiments/aggregate_results.py

echo Done.
