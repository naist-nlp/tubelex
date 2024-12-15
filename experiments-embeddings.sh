#!/bin/bash

#SBATCH -J embeddings
#SBATCH -A lang
#SBATCH -p lang_short
#SBATCH -c1

if [ -n "$1" ]
then
	tasks="$1"
	langs="$2"
else
	tasks="ana sim"
	langs="en es"
fi

sbatch -Jsim-en experiments-embeddings.sh sim en
sbatch -Jsim-es experiments-embeddings.sh sim es
sbatch -Jana-en experiments-embeddings.sh ana en
sbatch -Jana-es experiments-embeddings.sh ana es

results=experiments/embeddings
mkdir -p $results

for task in $tasks
do
	if [ "$task" = 'sim' ]
	then
		opt='--similarity'
	else
		opt=''
	fi
	for lang in $langs
	do
		echo $task $lang
		base="${results}/${task}-${lang}"
		python experiments/embeddings.py $opt $lang fasttext/tubelex-${lang}.vec					> "${base}-tubelex.tsv"
		python experiments/embeddings.py $opt $lang pretrained-fasttext/wiki.${lang}.vec			> "${base}-wiki.tsv"
		python experiments/embeddings.py $opt $lang pretrained-fasttext/subs.${lang}.1e6.clean.vec	> "${base}-opensub.tsv"
	done
done
