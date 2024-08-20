#!/bin/bash

mkdir -p fasttext

variant="$1"

case "$variant" in
	''|skipgram-100)
		variant="skipgram-100"
		params="skipgram"
		;;
	skipgram-300)
		# "Baseline" (Grave et al., 2018)
		params="skipgram -dim 300"
		;;
	cbow-300-hq)
		# The final model, except for character ngram length restriction (Grave et al., 2018)
		params="cbow -dim 300 -neg 10 -epoch 10"
		;;
	cbow-100-hq-fast)
		# The final model, but with dim=100 (Grave et al., 2018)
		params="cbow -dim 100 -neg 10 -epoch 10 -minn 5 -maxn 5"
		;;
	cbow-300-hq-fast)
		# The final model (Grave et al., 2018)
		params="cbow -dim 300 -neg 10 -epoch 10 -minn 5 -maxn 5"
		;;
	*)
		echo "Unknown vector type '$1', exiting." >&2
		exit 1
		;;
esac
