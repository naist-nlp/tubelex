from collections import defaultdict
import pandas as pd
import numpy as np
import os

TASK2NAME = {
    'sim': 'Word Similarity',
    'ana': 'Word Analogy'
    }

CORPUS2ID = {
    'Wikipedia': 'wiki',
    'OpenSubtitles': 'opensub',
    'TUBELEX': 'tubelex'
    }


def sort_by_corpus(df: pd.DataFrame) -> pd.DataFrame:
    corpora_order = list(CORPUS2ID.keys())
    return df.sort_index(
        axis=0,
        key=lambda cols: pd.Index(map(
            lambda col: corpora_order.index(col) if (col in corpora_order)
            else col,
            cols
            ))
        )

def sort_col_by_name(df: pd.DataFrame) -> pd.DataFrame:
    name_order = list(ROW2NAME.values())
    return df.sort_index(
        axis=1,
        key=lambda rows: pd.Index(map(
            lambda row: name_order.index(row) if (row in name_order)
            else row,
            rows
            ))
        )


LANG2ID = {
    'English': 'en',
    'Spanish': 'es'
    }

ROW2NAME = {
    'geography': 'Sem.: Geography',
    'family': 'Sem.: Family',
    'semantic': 'Semantic',
    'syntactic': 'Syntactic',
    'all': 'Total',
    'pearson': r"Pearson's $r$",
#    'pearson_p'
    'spearman': r"Spearman's $\rho$",
#    'spearman_p'
#    'oov_ratio': r'Missing'
    }


def main():
    for task in ('sim', 'ana'):

        d = {}

        for lang, lang_id in LANG2ID.items():
            for corpus, corpus_id in CORPUS2ID.items():
                path = f'experiments/embeddings/{task}-{lang_id}-{corpus_id}.tsv'
                if os.path.exists(path):
                    print(f'Reading {path}')
                    df = pd.read_table(path, index_col=0, header=None)
                    if task == 'sim':
                        assert df.loc['spearman_p', 1] == 0
                        assert df.loc['pearson_p', 1] == 0
                    d[(lang, corpus)] = df.rename(ROW2NAME).reindex(
                        ROW2NAME.values()
                        )[1]

        combined = pd.DataFrame(d).transpose().dropna(axis=1)
#          .reset_index(
#             names=['lang', 'corpus']
#             ).pivot(index='corpus', columns='lang').swaplevel(axis=1)

        combined = sort_col_by_name(sort_by_corpus(combined))

        combined.index.names = ['Language', 'Embeddings']
        combined.to_csv(f'experiments/embeddings/{task}.tsv', sep='\t',
                          float_format='%3f')


if __name__ == '__main__':
    main()
