from collections import defaultdict
import pandas as pd
import numpy as np
import os
from corrstats import dependent_corr


TUBELEX = 'TUBELEX\\textsubscript{default}'
CORPUS2ID = {
    'BNC-Spoken': 'spoken-bnc',
    'CREA-Spoken': 'alonso',
    'CSJ': 'csj-lemma',
    'HKUST/MTS': 'hkust-mtsc',

    'ACTIV-ES': 'activ-es',
    'EsPal': 'espal',
    'LaboroTV1+2': 'laborotv',
    'OpenSubtitles': 'os',
    'SubIMDB': 'subimdb',
    'SubIMDB_R': 'subimdb-regex',
    'SUBTLEX': 'subtlex',
    'SUBTLEX-UK': 'subtlex-uk',
    'SUBTLEX_R': 'subtlex-regex',


    'GINI': 'gini',
    'GINI_R': 'gini-regex',
    'Wikipedia': 'wiki',
    'Wikipedia_R': 'wiki-regex',
    'wordfreq': 'wordfreq',
    'wordfreq_R': 'wordfreq-regex',

    'TUBELEX\\textsubscript{default}': 'tubelex',
    'TUBELEX\\textsubscript{regex}': 'tubelex-regex',
    'TUBELEX\\textsubscript{base}': 'tubelex-base',
    'TUBELEX\\textsubscript{lemma}': 'tubelex-lemma',
    'TUBELEX_entertainment': 'tubelex-entertainment',
    'TUBELEX_entertainment_L': 'tubelex-entertainment-lemma',
    'TUBELEX_entertainment_B': 'tubelex-entertainment-base',
    'TUBELEX_entertainment_R': 'tubelex-entertainment-regex',
    'TUBELEX_comedy': 'tubelex-comedy',
    'TUBELEX_comedy_L': 'tubelex-comedy-lemma',
    'TUBELEX_comedy_B': 'tubelex-comedy-base',
    'TUBELEX_comedy_R': 'tubelex-comedy-regex',
    }

TASK2NAME = {
    'ldt': 'Decision Time',
    'fam': 'Familiarity',
    'fam-alt': 'Familiarity (Alternative Datasets)',
    'mlsp': 'Complexity',
    }

COL2ID = {
    'Pearson\'s r': 'correlation'
    }


def dependent_corr_pvalue_or_nan(
    xy, xz, yz, n, twotailed=True, conf_level=0.95, method='steiger'
    ) -> float:
    if np.isnan(xy) or np.isnan(xz) or np.isnan(yz) or np.isnan(n):
        return np.nan
    if (xy == xz) and (yz == 1):
        return np.nan
    p = dependent_corr(
        xy, xz, yz, n, twotailed=twotailed, conf_level=conf_level, method=method
        )[1]
    if np.isnan(p):
        # Technical/numerical thing: sometimes we get nans for very similar
        # correlation, we want to differentiate it from exactly the same correlation
        # and nan inputs above
        return 1
    return p


LANG2ALT_DESC = {
    'English': '(Glasgow)',
    'Spanish': '(Moreno-Martínez)',
    'Japanese': '(Amano+Kondo)'
    }


def main():
    # Unnecessary:
    # task_name2df = {}
    # task_name2df_p = {}
    for filename, cols, add_mlsp, task in (
        ('experiments/mlsp-results', ['R2', 'Pearson\'s r'], True, None),
        *((
            f'experiments/{task}-corr',
            ['correlation', 'corr_tubelex', 'n', 'n_missing', 'corr_without_missing'],
            False,
            task
            ) for task in ('ldt', 'fam', 'fam-alt', 'mlsp'))
            # Exclude 'ldtz' (LDT z-scores) : we have z-scores only for en and zh, and
            # the results are basically the same as for means ('ldt').
            ):

        d = defaultdict(dict)

        for corpus, corpus_id in CORPUS2ID.items():
            path = f'{filename}-{corpus_id}.tsv'
            if os.path.exists(path):
                print(f'Reading {path}')
                df = pd.read_table(path, index_col='language')
                for col in cols:
                    if col in df:
                        d[col][corpus] = df[col]

        if add_mlsp:
            for col in cols:
                col_id = COL2ID.get(col, col)
                print(f'Reading MLSP shared task data for {col_id}')
                mlsp = pd.read_table(f'{filename}-shared-task-{col_id}.tsv',
                                     index_col='language')
                for c in mlsp.columns:
                    d[col][c] = mlsp[c]

        combined_dfs = {}

        for col, data_dict in d.items():
            combined = pd.DataFrame(data_dict).transpose()

            if task == 'fam-alt':
                combined.columns = pd.MultiIndex.from_arrays([
                    combined.columns,
                    [LANG2ALT_DESC[lang] for lang in combined.columns]
                    ], names=None)

            col_id = COL2ID.get(col, col)
            combined.to_csv(f'{filename}-aggregate-{col_id}.tsv', sep='\t')
            combined_dfs[col] = combined

        if 'correlation' in cols:
            r_task_corp     = combined_dfs['correlation']
            r_task_tubelex  = r_task_corp.loc[TUBELEX]
            r_corp_tubelex  = combined_dfs['corr_tubelex']
            ns              = combined_dfs['n']
            d_pvalues       = {}
            for lang, rxt in r_task_tubelex.items():
                # rxt is a single number
                lang_r_task_corp    = r_task_corp[lang]
                lang_r_corp_tubelex = r_corp_tubelex[lang]
                lang_n              = ns[lang]
                d_pvalues[lang] = [
                    dependent_corr_pvalue_or_nan(rxc, rxt, rct, n)
                    for rxc, rct, n
                    in zip(lang_r_task_corp, lang_r_corp_tubelex, lang_n)
                    ]
            df_pvalues = pd.DataFrame(d_pvalues, index=r_task_corp.index)
            df_pvalues.to_csv(f'{filename}-aggregate-pvalues.tsv', sep='\t',
                              float_format='%4f')

            # unnecessary
            # task_name = TASK2NAME[task]
            # task_name2df[task_name] = combined_dfs['correlation']
            # task_name2df_p[task_name] = df_pvalues

    # unnecessary
    # df = pd.concat(task_name2df.values(), axis=1, keys=task_name2df.keys())
    # df_p = pd.concat(task_name2df_p.values(), axis=1, keys=task_name2df_p.keys())
    #
    # df.to_csv(f'all-aggregate-correlation.tsv', sep='\t',
    #                   float_format='%4f')
    # df_pvalues.to_csv(f'all-aggregate-pvalues.tsv', sep='\t',
    #                   float_format='%4f')


if __name__ == '__main__':
    main()
