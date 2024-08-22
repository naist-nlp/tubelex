#!/bin/bash


#TODO remove from production:
# Speed things up
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
	python experiments/run.py $var_opt --tubelex $langs --corr    --mlsp $mlsp > experiments/mlsp-corr-tubelex${variant}.tsv
	
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

echo activ-es
python experiments/run.py --activ-es --train   --mlsp spanish_lcp_labels
python experiments/run.py --activ-es --metrics --mlsp spanish_lcp_labels > experiments/mlsp-results-activ-es.tsv
python experiments/run.py --activ-es --corr    --mlsp spanish_lcp_labels > experiments/mlsp-corr-activ-es.tsv


for variant in '' -regex
do
	case "$variant" in
	  '')
		langs="es en ja"
		mlsp="$all"	
		var_opt=''
		;;
	  -regex)
		langs="es en"
		mlsp="$esen"
		var_opt='--tokenization regex'
		;;
	esac
	if [ -z "$variant" ]
	then
		# We always use the wordfreq tokenization for wordfreq
		echo wordfreq$variant
		python experiments/run.py $var_opt --wordfreq $langs --train   --mlsp $mlsp
		python experiments/run.py $var_opt --wordfreq $langs --metrics --mlsp $mlsp > experiments/mlsp-results-wordfreq${variant}.tsv
		python experiments/run.py $var_opt --wordfreq $langs --corr    --mlsp $mlsp > experiments/mlsp-corr-wordfreq${variant}.tsv
	fi
	
done
echo wiki
python experiments/run.py --wiki es en ja --train   --mlsp $all
python experiments/run.py --wiki es en ja --metrics --mlsp $all > experiments/mlsp-results-wiki.tsv
python experiments/run.py --wiki es en ja --corr    --mlsp $all > experiments/mlsp-corr-wiki.tsv
	
for variant in '' -regex
do
	case "$variant" in
	  '')
		langs="en ja"
		mlsp="english_lcp_labels japanese_lcp_labels"	
		var_opt=''
		;;
	  -regex)
		langs="en"
		mlsp="english_lcp_labels"	
		var_opt='--tokenization regex'
		;;
	esac
	echo gini$variant
	python experiments/run.py $var_opt --minus --gini $langs --train   --mlsp $mlsp
	python experiments/run.py $var_opt --minus --gini $langs --metrics --mlsp $mlsp > experiments/mlsp-results-gini${variant}.tsv
	python experiments/run.py $var_opt --minus --gini $langs --corr    --mlsp $mlsp > experiments/mlsp-corr-gini${variant}.tsv
done

# Doesn't improve:
# echo GINI-log
# python experiments/run.py --minus-log --gini en ja --train  \
# 	english_lcp_labels japanese_lcp_labels
# python experiments/run.py --minus-log --gini en ja --metrics \
# 	english_lcp_labels japanese_lcp_labels > experiments/mlsp-results-gini-log.tsv
# python experiments/run.py --minus-log --gini en ja --corr \
# 	english_lcp_labels japanese_lcp_labels > experiments/mlsp-corr-gini-log.tsv



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
	python experiments/run.py $var_opt --subtlex es en --train   --mlsp $esen
	python experiments/run.py $var_opt --subtlex es en --metrics --mlsp $esen > experiments/mlsp-results-subtlex${variant}.tsv
	python experiments/run.py $var_opt --subtlex es en --corr    --mlsp $esen > experiments/mlsp-corr-subtlex${variant}.tsv
done

echo subimdb
python experiments/run.py --subimdb --train   --mlsp english_lcp_labels
python experiments/run.py --subimdb --metrics --mlsp english_lcp_labels   > experiments/mlsp-results-subimdb.tsv
python experiments/run.py --subimdb --corr    --mlsp english_lcp_labels   > experiments/mlsp-corr-subimdb.tsv


echo
echo '==='
echo 'LDT'
echo '==='
echo
for corpus in wordfreq tubelex tubelex-entertainment tubelex-comedy wiki subtlex os
do
	if [[ "$corpus" =~ - ]]
	then
		corpus_opt="--cat ${corpus#*-} --${corpus%-*}"
	else
		corpus_opt="--${corpus}"
	fi
	echo "$corpus"
	python experiments/run.py $corpus_opt     en es zh                   --corr --ldt en es zh > experiments/ldt-corr-${corpus}.tsv
	if [[ "$corpus" =~ ^tubelex ]]
	then
		python experiments/run.py $corpus_opt en es --form lemma         --corr --ldt en es    > experiments/ldt-corr-${corpus}-lemma.tsv
		python experiments/run.py $corpus_opt en es --tokenization regex --corr --ldt en es    > experiments/ldt-corr-${corpus}-regex.tsv
	elif [[ "$corpus" != 'wordfreq' ]] && [[ "$corpus" != 'wiki' ]]
	then
		# We always use wordfreq tokenization for wordfreq:
		python experiments/run.py $corpus_opt en es --tokenization regex --corr --ldt en es    > experiments/ldt-corr-${corpus}-regex.tsv
	fi
done
# Regex wouldn't change anything for en:
echo "gini"
python experiments/run.py --minus --gini en                              --corr --ldt en       > experiments/ldt-corr-gini.tsv
echo "subimdb"
python experiments/run.py --subimdb 				                     --corr --ldt en       > experiments/ldt-corr-subimdb.tsv
echo "espal"
python experiments/run.py --espal 										 --corr --ldt es > 
echo "activ-es"
python experiments/run.py --activ-es 									 --corr --ldt es > experiments/ldt-corr-activ-es.tsv



echo
echo '==========='
echo 'LDT z score'
echo '==========='
echo
for corpus in tubelex subtlex os
do
	corpus_opt="--${corpus}"
	echo "$corpus"
	python experiments/run.py $corpus_opt     en zh                   --corr -z --ldt en zh 	  > experiments/ldtz-corr-${corpus}.tsv
done
echo "subimdb"
python experiments/run.py --subimdb 				                     --corr -z --ldt en       > experiments/ldtz-corr-subimdb.tsv


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
echo subtlex
python experiments/run.py --subtlex zh                                --corr --fam zh    > experiments/fam-corr-subtlex.tsv

rm -r experiments/models experiments/output experiments/output.tsv

echo Aggregating

python experiments/aggregate_results.py

echo Done.
