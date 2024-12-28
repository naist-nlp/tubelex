import argparse
from collections import defaultdict
import pandas as pd
import numpy as np
import os
from corrstats import dependent_corr


TUBELEX = 'TUBELEX\\textsubscript{default}'
GINI = 'GINI'
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

MEASURES = [
    'frequency',
    #'simple_frequency', same as frequency => breaks pvalue computation
    'range_videos',
    'range_channels',
    'range_categories',
    'weighted_range',
    'gini',
    'maxmin',
    'ada',
    'juilland_d',
    'vmr',
    'gries_dp',
    'gries_dp_eq',
    'rosengren_s',
    'sqrt',
    'carrol_d2'
    ]
TRANSFORMS = ['', 'log_']   # NO IMPROVEMENT: 'sqrt_'
MEASURE2ID = { # TODO
    tm: f'tubelex-{tm}' for t in TRANSFORMS for m in MEASURES for tm in (t + m,)
    }

TASK2NAME = {
    'ldt': 'Decision Time',
    'fam': 'Familiarity',
    'fam-c-gini': 'Familiarity (compared with GINI)',
    'fam-alt': 'Familiarity (Alternative Datasets)',
    'mlsp': 'Complexity',
    }

MEASURES_TASKS = ['ldt', 'fam', 'mlsp']

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--measures', action='store_true',
                        help='Aggregate expriments with dispersion measures.')
    return parser.parse_args()


def main(args: argparse.Namespace):
    # Unnecessary:
    # task_name2df = {}
    # task_name2df_p = {}

    if args.measures:
        data_to_aggregate = (
            # TODO ('experiments/mlsp-results', ['R2', 'Pearson\'s r'], True, None),
            *((
                f'experiments/measures-{task}-corr',
                ['correlation',
                 'adjusted_r2',
                 'corr_tubelex',
                 'n', 'n_missing',
                 'corr_without_missing'
                 ],
                False,
                task
                ) for task in MEASURES_TASKS),
            )
        method2id = MEASURE2ID
    else:
        data_to_aggregate = (
            ('experiments/mlsp-results', ['R2', 'Pearson\'s r'], True, None),
            *((
                f'experiments/{task}-corr',
                ['correlation',
                 'corr_gini' if task == 'fam-c-gini' else 'corr_tubelex',
                 'n', 'n_missing',
                 'corr_without_missing'
                 ],
                False,
                task
                ) for task in TASK2NAME)
            # Exclude 'ldtz' (LDT z-scores) : we have z-scores only for en and zh, and
            # the results are basically the same as for means ('ldt').
            )
        method2id = CORPUS2ID

    for filename, cols, add_mlsp, task in data_to_aggregate:

        d = defaultdict(dict)

        for method, method_id in method2id.items():
            path = f'{filename}-{method_id}.tsv'
            if os.path.exists(path):
                print(f'Reading {path}')
                df = pd.read_table(path, index_col='language')
                for col in cols:
                    if col in df:
                        d[col][method] = df[col]

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
            if 'corr_tubelex' in cols:
                corr2_col = 'corr_tubelex'
                if args.measures:
                    corp2 = 'log_frequency'
                else:
                    corp2 = TUBELEX
            else:
                assert 'corr_gini' in cols, (filename, cols)
                corr2_col = 'corr_gini'
                corp2 = GINI
            r_task_corp     = combined_dfs['correlation']
            r_task_tubelex  = r_task_corp.loc[corp2]
            r_corp_tubelex  = combined_dfs[corr2_col]
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
    main(parse_args())
