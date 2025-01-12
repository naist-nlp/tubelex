import argparse
from collections import defaultdict
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

#
# MEASURES = [
#     'frequency',
#     'range',
#     'weighted_range',
#     'range_nofreq',
#     'range_nofreq_gries',
#     'sort_gini',
#     'maxmin',
#     'juilland_d',
#     'vmr',
#     'gries_dp',
#     'rosengren_s',
#     'carrol_d2'
#     ]
# TRANSFORMS = ['', 'log_']   # NO IMPROVEMENT: 'sqrt_'
# VARIANTS = ['', '_channels', '_videos']
# MEASURE2ID = { # TODO
#     tm: f'tubelex-{tm}'
#     for t in TRANSFORMS
#     for v in VARIANTS
#     for m in MEASURES for tm in (t + m + v,)
#     }
#
# TASK2NAME = {
#     'ldt': 'Decision Time',
#     'fam': 'Familiarity',
#     'fam-c-gini': 'Familiarity (compared with GINI)',
#     'fam-alt': 'Familiarity (Alternative Datasets)',
#     'mlsp': 'Complexity',
#     }
#
MEASURES_TASKS = ['ldt', 'fam', 'mlsp']
#
# COL2ID = {
#     'Pearson\'s r': 'correlation'
#     }
#

TASK2R_FILES = {
    task: f'experiments/measures-{task}-corr-aggregate-correlation.tsv'
    for task in MEASURES_TASKS
    }
TASK2P_FILES = {
    task: f'experiments/measures-{task}-corr-aggregate-pvalues.tsv'
    for task in MEASURES_TASKS
    }
TASK2R2_FILES = {
    task: f'experiments/measures-{task}-corr-aggregate-adjusted_r2.tsv'
    for task in MEASURES_TASKS
    }
TASK2N_FILES = {
    task: f'experiments/measures-{task}-corr-aggregate-n.tsv'
    for task in MEASURES_TASKS
    }

LOGF = 'log_frequency'

def r_n2adjusted_r2(
    r: pd.DataFrame,
    n: pd.DataFrame  # numbers of examples
    ) -> pd.DataFrame:
    # Mordecai Ezekiel (1930) "Methods of correlation analysis"
    # We assume single independent variable, hence (n - p - 1) = (n - 2).
    return 1 - (1 - r ** 2) * (n - 1) / (n - 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', '-a', action='store_true',
                        help='List all instead of best.')
    parser.add_argument('--no-select-log', '-l', action='store_true',
                        help='Do not select log/non-log variants.')
    return parser.parse_args()

def my_round(x):
    return x.round(3)

def metric2transform_base_parts(m: str):
    transform = ''
    if m.startswith('log_'):
        m = m.removeprefix('log_')
        transform = 'log'
    if m == 'frequency':
        parts = 'tokens'
    elif m.endswith('_videos'):
        m = m.removesuffix('_videos')
        parts = 'videos'
    elif m.endswith('_channels'):
        m = m.removesuffix('_channels')
        parts = 'channels'
    else:
        parts = 'categories'
    return (transform, m, parts)

MEASURE2NAME = {
    'range':        'Range',
    'sort_gini':    'Gini Index',
    'juilland_d':   'Juilland\'s $D$',
    'gries_dp':     'Gries\'s DP',
    'rosengren_s':  'Rosengren\'s $S$',
    'carrol_d2':    'Carroll\'s $D_2$',
    'lyne_d3':      'Lyne\'s $D_3$',
    'frequency':    'Frequency'
    }
PART_SUF2NAME = {
    '_videos': 'Videos',
    '_channels': 'Channels',
    '': 'Categories'
    }


def print_log_tables(r2_wo_logf, r2_w_logf):
    print()
    print()
    print('=========== LOG TABLES (to be edited for paper) =========')

    for name, df in (
        ('wo', r2_wo_logf),
        ('w', r2_w_logf)
        ):
        eval_log = {part: [] for part in PART_SUF2NAME.values()}
        for m_base in MEASURE2NAME:
            for part_suf, part in PART_SUF2NAME.items():
                if m_base == 'frequency':
                    if name == 'wo':
                        if part_suf:
                            eval_log[part].append('')
                            continue
                        # else: add the usual value
                    else:
                        continue
                m = m_base + part_suf
                deltas = df.loc['log_' + m] - df.loc[m]
                nbetter = (deltas > 0).sum()
                mean_d = deltas.mean()
                pos = mean_d >= 0.001
                v = (
                    rf'$\phantom{{-}}\mathbf{{{mean_d:.3f}}}$' if pos else
                    rf'${mean_d:.3f}$'
                    )
                if nbetter == 0:
                    v = rf'{v}\phantom{{ (00)}}'
                elif nbetter < 10:
                    v = rf'{v} ({nbetter})\phantom{{0}}'
                else:
                    v = rf'{v} ({nbetter})'
                eval_log[part].append(v)
        idx = list(MEASURE2NAME.values())
        if name == 'w':
            idx = idx[:-1]
        df = pd.DataFrame(eval_log, index=idx)
        print(rf'\label{{tab:log{name}}}')
        print(df.to_latex(column_format='lccc'))

def select_log_non_log(df: pd.DataFrame, to_log: set[str]):
    idx     = df.index.to_series()
    is_log  = idx.str.startswith('log_')
    do_log  = idx.str.removeprefix('log_').apply(to_log.__contains__)
    return df[is_log == do_log]

def measure_part_df_frequency_idx(df: pd.DataFrame, to_log: set[str], index=None):
    mean = df.mean(axis=1)
    mean = select_log_non_log(mean, to_log)
    classif = mean.index.to_series()

    mp_mean = pd.DataFrame.from_records(
        classif.apply(metric2transform_base_parts).to_list(),
        columns=['m_trans', 'm_base', 'm_parts'],
        index=classif.index
        )
    mp_mean['mean'] = mean
    # parts_wo_mean_r2 = parts_wo_mean_r2.drop(columns=['m_trans'])
    mp_mean = mp_mean.drop('log_frequency').pivot(columns='m_parts', index='m_base', values='mean')
    mp_mean['max'] = mp_mean.max(axis=1)

    if index is not None:
        mp_mean = mp_mean.reindex(index=index)
    else:
        mp_mean = mp_mean.sort_values(by='max')
    mp_mean = mp_mean[[
        'categories', 'channels', 'videos'
        ]]
    return (mp_mean, mean.loc['log_frequency'], mp_mean.index)

def main(args: argparse.Namespace):

    font_size = 10.5
    plt.rcParams.update({'font.size': font_size})


    # PCC (r) for a single variable (in index)
    task2r = {task: pd.read_table(f, index_col=0) for task, f in TASK2R_FILES.items()}
    # P-values for the difference between correlation with log frequency and the
    # variable in index
    task2p = {task: pd.read_table(f, index_col=0) for task, f in TASK2P_FILES.items()}
    # Numbers of examples:
    task2n = {task: pd.read_table(f, index_col=0) for task, f in TASK2N_FILES.items()}
    # Adjusted R2 for two variables (the one in index + log frequency):
    task2r2 = {task: pd.read_table(f, index_col=0) for task, f in TASK2R2_FILES.items()}

    # columns will be (task, language):
    r = pd.concat(task2r, axis=1)
    p = pd.concat(task2p, axis=1)
    n_examples = pd.concat(task2n, axis=1)
    r2_w_logf = pd.concat(task2r2, axis=1)
    r2_wo_logf = r_n2adjusted_r2(r, n_examples)
    delta_r2_w_logf = r2_w_logf - r2_w_logf.loc[LOGF]
    delta_r2_wo_logf = r2_wo_logf - r2_wo_logf.loc[LOGF]

    n_data = len(r.columns)
    wo_sig_stronger         = (delta_r2_wo_logf > 0) & (p < 0.001)
    wo_not_sig_different    = (p >= 0.001)
    wo_not_sig_weaker       = wo_sig_stronger | wo_not_sig_different
    wo_tad_stronger         = (delta_r2_wo_logf > 0.01)
    wo_not_too_weak         = (delta_r2_wo_logf >= -0.01)
    wo_mean_delta_r2        = delta_r2_wo_logf.mean(axis=1)


    w_tad_stronger          = (delta_r2_w_logf > 0.01)
    w_not_too_weak          = (delta_r2_w_logf >= -0.01)
    w_mean_delta_r2         = delta_r2_w_logf.mean(axis=1)



    # This is in line with the results in `print_log_tables()`
    w_log_measures = {
        'range_videos', 'range_channels',
        'sort_gini_videos', 'sort_gini_channels',
        'rosengren_s_videos', 'rosengren_s_channels',
        'frequency'}
    wo_log_measures = {*w_log_measures, 'sort_gini_categories'}

    # never significantly worse than log f, never worse more than by 0.01
    wo_good_measures = {'range_videos', 'range_channels'}
    # better by 0.01 for at least 8/11 datasets:
    w_good_measures = {'rosengren_s_categories', 'range_categories', 'range_videos', 'range_channels'}


    wo_scores = pd.DataFrame({
        ('mean', 'ALL'): my_round(wo_mean_delta_r2),
        ('mean', 'fam'): my_round(delta_r2_wo_logf['fam'].mean(axis=1)),
        ('mean', 'ldt'): my_round(delta_r2_wo_logf['ldt'].mean(axis=1)),
        ('mean', 'mlsp'): my_round(delta_r2_wo_logf['mlsp'].mean(axis=1)),
        ('mean', 'en'): my_round(delta_r2_wo_logf.xs('English', axis=1, level=1).mean(axis=1)),
        ('mean', 'ja'): my_round(delta_r2_wo_logf.xs('Japanese', axis=1, level=1).mean(axis=1)),
        ('mean', 'es'): my_round(delta_r2_wo_logf.xs('Spanish', axis=1, level=1).mean(axis=1)),
        ('mean', 'id'): my_round(delta_r2_wo_logf.xs('Indonesian', axis=1, level=1).mean(axis=1)),
        ('mean', 'zh'): my_round(delta_r2_wo_logf.xs('Chinese', axis=1, level=1).mean(axis=1)),
        ('p_strict', 'ALL'): wo_sig_stronger.sum(axis=1),
        ('strict', 'ALL'): wo_tad_stronger.sum(axis=1),
        ('p_relaxed', 'ALL'): wo_not_sig_weaker.sum(axis=1),
        ('relaxed', 'ALL'): wo_not_too_weak.sum(axis=1),
        }).sort_values(by=('mean', 'ALL'))
    if not args.all:
        wo_scores = wo_scores[wo_scores['p_strict', 'ALL'] > 0]
    if not args.no_select_log:
        wo_scores = select_log_non_log(wo_scores, wo_log_measures)
    print(wo_scores.to_string())



    print()
    print('WITH LOG F')
    w_scores = pd.DataFrame({
        ('mean', 'ALL'): my_round(w_mean_delta_r2),
        ('mean', 'fam'): my_round(delta_r2_w_logf['fam'].mean(axis=1)),
        ('mean', 'ldt'): my_round(delta_r2_w_logf['ldt'].mean(axis=1)),
        ('mean', 'mlsp'): my_round(delta_r2_w_logf['mlsp'].mean(axis=1)),
        ('mean', 'en'): my_round(delta_r2_w_logf.xs('English', axis=1, level=1).mean(axis=1)),
        ('mean', 'ja'): my_round(delta_r2_w_logf.xs('Japanese', axis=1, level=1).mean(axis=1)),
        ('mean', 'es'): my_round(delta_r2_w_logf.xs('Spanish', axis=1, level=1).mean(axis=1)),
        ('mean', 'id'): my_round(delta_r2_w_logf.xs('Indonesian', axis=1, level=1).mean(axis=1)),
        ('mean', 'zh'): my_round(delta_r2_w_logf.xs('Chinese', axis=1, level=1).mean(axis=1)),
        ('strict', 'ALL'): w_tad_stronger.sum(axis=1),
        ('strict', 'fam'): w_tad_stronger['fam'].sum(axis=1),
        ('strict', 'ldt'): w_tad_stronger['ldt'].sum(axis=1),
        ('strict', 'mlsp'): w_tad_stronger['mlsp'].sum(axis=1),
        ('relaxed', 'ALL'): w_not_too_weak.sum(axis=1),
        }).sort_values(by=('mean', 'ALL'))
    if not args.all:
        w_scores = w_scores[w_scores['strict', 'ALL']>n_data/2]
    print(w_scores.to_string())


    parts_wo_mean_r2, baseline_y, idx = measure_part_df_frequency_idx(r2_wo_logf, wo_log_measures)
    # TESTING
    # r2_w_logf.loc['sort_gini_videos',:] = 0.5
    # r2_w_logf.loc['log_range_videos',:] = 0.45
    # r2_w_logf.loc['log_range_channels',:] = 0.45
    parts_w_mean_r2, *_ = measure_part_df_frequency_idx(r2_w_logf, w_log_measures, index=idx)

    # original NLP 2025 paper: plt.figure(figsize=(11, 5))

    plt.figure(figsize=(9, 5))

    palette = sns.color_palette('viridis')

    barw = 0.75

    parts_wo_mean_r2_display = parts_wo_mean_r2.rename(columns=str.capitalize)
    parts_wo_mean_r2_display = parts_wo_mean_r2_display.rename(index=MEASURE2NAME)

    parts_w_mean_r2_display = parts_w_mean_r2.rename(columns=str.capitalize)
    parts_w_mean_r2_display = parts_w_mean_r2_display.rename(index=MEASURE2NAME)

    ax = parts_wo_mean_r2_display.plot(
        kind='bar', stacked=False, ax=plt.gca(), color=palette[::-2],
        width=barw
        )

    parts_w_mean_r2_display.plot(
        kind='bar', stacked=False, ax=plt.gca(), color=palette[::-2],
        # White hatching instead of alpha=0.5:
        edgecolor='white', linewidth=0, hatch_linewidth=1.5, hatch='//////',
        zorder=-1,
        width=barw
        )

    # Y axis:
    plt.gca().set_ylim([0, 0.615])    # nicer and labels fit
    plt.yticks(np.arange(0, 0.51, 0.1), rotation='horizontal')

    # X axis:
    plt.xticks(rotation='horizontal', va='baseline', y=-0.02)   # align to baseline

    # Baseline:
    plt.axhline(y=baseline_y, linestyle='--', linewidth=1, color=palette[0])
    plt.text(
        x=-0.52,
        y=baseline_y - 0.025,  # Slightly below the baseline
        s=f'log-frequency: {baseline_y:.3f}',
        fontsize=font_size,
        ha='left'
    )

    # Extra annotation:
    for i, m in enumerate(parts_w_mean_r2.index):
        for j, p in enumerate(parts_w_mean_r2.columns):
            x = i + (j - 1) / len(parts_w_mean_r2.columns) * barw

            # W:
            y = parts_w_mean_r2.loc[m, p]
            s = f'{y:.3f}'
            if f'{m}_{p}' in w_log_measures:
                s += ' (log)'
            if f'{m}_{p}' in w_good_measures:
                s += ' ★'   # star
            plt.text(x, y+0.01, s=s, fontsize=font_size, color='k', ha='center',
                     rotation='vertical')
#             if f'{m}_{p}' in wo_log_measures:
#                 # plt.plot(x, y+0.0125, '*', markersize=8, color='r')
#                 plt.text(
#                     x, y-0.005, s='log', fontsize=font_size, color='k',
#                     rotation='vertical', ha='center'
#                     )

            # WO:
            y = parts_wo_mean_r2.loc[m, p]
            s = f'{y:.3f}'
            if f'{m}_{p}' in wo_log_measures:
                s += ' (log)'
            if f'{m}_{p}' in wo_good_measures:
                s += ' ★'
            plt.text(x, y-0.005, s=s, fontsize=font_size, color='w',
                     rotation='vertical', ha='center', va='top'
                     )

    # Add labels and title
    plt.xlabel('Dispersion Measure (DM)')
    plt.ylabel(r'$R_\text{a}^2$ (Mean Over 11 Datasets)') # $\overline{R_\text{a}^2}$

    handles, labels = ax.get_legend_handles_labels()
    legend1 = plt.legend(
        handles[:3], labels[:3],
        title='Single Variable (DM)', alignment='left',
        bbox_to_anchor=(0.7, 0.0, 0.3, 0.0), loc='lower left', mode='expand',
        #loc='center right',
        framealpha=1
        )
    plt.legend(
        handles[3:], labels[3:],
        title='Two Variables (DM, log-freq.)', alignment='left',
        bbox_to_anchor=(0.7, 0.23, 0.3, 0.0), loc='lower left', mode='expand',
        # bbox_to_anchor=(0, 0.575), loc='lower left',     # under legend1
        framealpha=1
        )
    plt.gca().add_artist(legend1)
    plt.tight_layout()

    # Show the plot
    # plt.show()
    plt.savefig('experiments/figures/dispersion_adj_r2.pdf', bbox_inches='tight')


    print_log_tables(r2_wo_logf, r2_w_logf)


#     print()
#     print()
#     print('IDXMAX')
#     print(delta_r2_wo_logf.idxmax())
#     print()
#     print()
#
#     for col, x in delta_r2_wo_logf.items():
#         top = x.sort_values(ascending=False)
#         ouridx = top.index.get_loc('log_range_channels')
#         if ouridx:
#             print(col, list(zip(top.index[:ouridx+1].to_list(), top.iloc[:ouridx+1].to_list())))
#

#
#     for col, x in delta_r2_w_logf.items():
#         top = x.sort_values(ascending=False)
#         ouridx = top.index.get_loc('log_range_channels')
#         if ouridx:
#             print(col, list(zip(top.index[:ouridx+1].to_list(), top.iloc[:ouridx+1].to_list())))


    print(n_data)


#     print(len(r_not_sig_different.columns))
#     print(r_sig_stronger.sum(axis=1).sort_values())
#     print()
#     print(r_tad_stronger.sum(axis=1).sort_values())
#
#     print(
#         (r_sig_stronger.sum(axis=1) >= r_tad_stronger.sum(axis=1)).all()
#         )

#     print()
#     print(r_not_sig_different.sum(axis=1).sort_values())
#     print()
#     print(r_not_sig_weaker.sum(axis=1).sort_values())
#
#
#     print()
#     print(np.nanmin(delta_r2_w_logf[r_sig_stronger].values),
#           np.nanmax(delta_r2_w_logf[r_sig_stronger].values))
#     print(np.nanmin(delta_r2_w_logf[r_not_sig_different].values),
#           np.nanmax(delta_r2_w_logf[r_not_sig_different].values))

    # print(delta_r2_wo_logf)





if __name__ == '__main__':
    main(parse_args())
