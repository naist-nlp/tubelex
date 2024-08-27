from functools import partial
import pandas as pd
import numpy as np
from pandas.io.formats.style import Styler  # type: ignore
from pandas.io.formats.style_render import Subset
import re

RESULT_PREC = 3
RESULT_TOL = 0.00001  # We use 4 decimal places in the original data
STAT_PREC = 0

GROUP2NAME = {
    0: r'speech',
    1: r'film/TV subtitles',
    2: r'other',
    3: r'our\vphantom{l}'    # alignment with previous group names after rotation
    }

GROUP2NAME_SMALL = {    # TODO now unnecessary DELETEME
    0: r'\scriptsize speech',
    1: r'film/TV subtitles',
    2: r'other',
    3: r'our\vphantom{l}'    # alignment with previous group names after rotation
    }

VALID_CORPORA2GROUP = {
    'BNC-Spoken': 0,
    'CREA-Spoken': 0,
    'CSJ': 0,
    'HKUST/MTS': 0,
    'ACTIV-ES': 1,
    'EsPal': 1,
    'LaboroTV1+2': 1,
    'OpenSubtitles': 1,
    'SubIMDB': 1,
    'SUBTLEX': 1,
    'SUBTLEX-UK': 1,
    'GINI': 2,
    'Wikipedia': 2,
    'wordfreq': 2,
    'TUBELEX\\textsubscript{default}': 3,
    'TUBELEX\\textsubscript{regex}': 3,
    'TUBELEX\\textsubscript{base}': 3,
    'TUBELEX\\textsubscript{lemma}': 3,
    }

# def _add_group_separators_by_line(tex: Sequence[str]) -> Iterator[str]:
#     group = 0
#     for line in tex:
#         line_group = next(
#             (g for c, g in VALID_CORPORA2GROUP.items() if c in line),
#             0   # ignore
#             )
#         if line_group > group:
#             group = line_group
#             yield '\midrule[0.025em]'
#         yield line
#
#
# def add_group_separators(tex: str) -> str:
#     return'\n'.join(_add_group_separators_by_line(tex.split('\n')))


def add_group_level(df: pd.DataFrame, small: bool = False) -> pd.DataFrame:
    group2name = GROUP2NAME_SMALL if small else GROUP2NAME
    return df.set_index(pd.MultiIndex.from_arrays(
        [
            pd.Series(map(
                lambda corpus: (
                    # rotate 90 deg., then make thin
                    r'\makebox[6pt][l]{\rotatebox[origin=c]{90}{%s}}' %
                    group2name[VALID_CORPORA2GROUP[corpus]]
                    ),
                df.index)),
            df.index
            ], names=None
        ))


def kill_extra_cline(tex: str) -> str:
    return re.sub(r'\\cline{[0-9-]+}\n\\bottomrule\n', r'\\bottomrule\n', tex)


def p_value2prop(p):
    if np.isnan(p):
        stars = '-'
    elif p < 0.001:
        stars = '***'
    elif p < 0.01:
        stars = '**'
    elif p < 0.05:
        stars = '*'
    else:
        stars = ''
    return f'pstars:{{{stars}}} --latex--rwrap;'


NP_P_VALUE2PROP = np.vectorize(p_value2prop)


def _add_p_stars(
    df: pd.DataFrame, p_values, pad_right=False, pad_top=False
    ) -> np.ndarray:
    prop = NP_P_VALUE2PROP(p_values)
    dfs = df.shape
    pvs = p_values.shape
    pad0 = dfs[0] - pvs[0]
    pad1 = dfs[1] - pvs[1]
    assert (pad0 == 0) or (pad0 >= 0 and pad_top)
    assert (pad1 == 0) or (pad1 >= 0 and pad_right)
    if pad_top or pad_right:
        prop = np.pad(
            prop,
            ((pad0, 0), (0, pad1)),
            mode='constant', constant_values=''
            )
    return prop


def add_p_stars(
    s: Styler,
    p_values: pd.DataFrame,
    pad_right=False,
    pad_top=False
    ) -> Styler:
    return s.apply(
        _add_p_stars, axis=None, p_values=p_values,
        pad_right=pad_right, pad_top=pad_top
        )


def _highlight_num_value(
    data: pd.DataFrame | pd.Series,
    op: str,
    props: str,
    tol: float = RESULT_TOL  # We use at mos 3 decimal places, so its appropriate
    ) -> np.ndarray:
    """
    Like pandas' _highlight_value(), but non-numeric strings are considered nan
    https://github.com/pandas-dev/pandas/blob/v2.1.0/pandas/io/formats/
    style.py#L3245C6-L3245C6
    """
    data = pd.to_numeric(data, errors='coerce')
    value = getattr(data, op)(skipna=True)
    if isinstance(data, pd.DataFrame):  # min/max must be done twice to return scalar
        value = getattr(value, op)(skipna=True)
    cond = np.abs(data - value) < tol
    # cond = data == value
    cond = cond.where(pd.notna(cond), False)
    return np.where(cond, props, '')


def highlight_extreme(
    s: Styler,
    props: str,
    subset: Subset | None = None,
    axis=0,
    op: str = 'max'
    ) -> Styler:
    return s.apply(
        partial(_highlight_num_value, op=op),
        axis=axis,
        subset=subset,
        props=props
        )


def background_gradient(
    s: Styler,
    index_subset: Subset | None = None,
    bool_subset: Subset | None = None,
    cmap='PuBu',
    low=0, high=0,
    axis=0,  # by column
    text_color_threshold=0.408,
    vmin=None,
    vmax=None,
    gmap=None
    ) -> Styler:
    '''
    Provides more finegrained masking via index subset + optional bool subset (mask).
    '''
    if bool_subset is None:
        return s.background_gradient(
            cmap=cmap, subset=(index_subset, slice(None)),
            low=low, high=high, axis=axis,
            text_color_threshold=text_color_threshold, vmin=vmin, vmax=vmax, gmap=gmap
            )
    assert index_subset is None
    assert axis is not None
    for idx in (s.data.index if axis else s.data.columns):
        bool_at_idx = (bool_subset.loc[idx] if axis else bool_subset[idx])
        gmap_at_idx = (gmap.loc[idx] if axis else gmap[idx])
        s = s.background_gradient(
            cmap=cmap, subset=(
                (idx, bool_at_idx) if axis else
                (bool_at_idx, idx)
                ),
            low=low, high=high, axis=axis,
            text_color_threshold=text_color_threshold, vmin=vmin, vmax=vmax,
            gmap=gmap_at_idx
            )
    return s


def limit_to_valid_corpora(df: pd.DataFrame) -> pd.DataFrame:
    valid = [c in VALID_CORPORA2GROUP for c in df.index]
    return df.loc[valid]


def alternate_row_bg(df: pd.DataFrame | pd.Series):
    s = np.arange(len(df)) % 2 != 0
    if isinstance(df, pd.Series):
        return np.where(s, 'cellcolor[HTML]:{EEEEEE}', '')
    s = pd.concat([pd.Series(s)] * df.shape[1], axis=1)
    z = pd.DataFrame(np.where(s, 'cellcolor[HTML]:{EEEEEE}', ''),
                     index=df.index, columns=df.columns)
    return z


def main():
    for results_id, best, header_levels in (
        ('mlsp', 'min', 1),
        ('ldt', 'min', 1),
        ('fam', 'max', 1),
        ('fam-alt', 'max', 2),
        ):
        tsv_header = list(range(header_levels))
        r = add_group_level(limit_to_valid_corpora(pd.read_table(
            f'experiments/{results_id}-corr-aggregate-correlation.tsv',
            index_col=0, header=tsv_header
            )))  # TODO DELETEME , small=(results_id == 'fam-alt') also in p
        p = add_group_level(limit_to_valid_corpora(pd.read_table(
            f'experiments/{results_id}-corr-aggregate-pvalues.tsv',
            index_col=0, header=tsv_header
            )))

        s = Styler(r, precision=RESULT_PREC, na_rep='---')

        r_not_na = ~r.isna()
        s = background_gradient(
            s,
            cmap='Blues', bool_subset=r_not_na,
            gmap=(-r if (best == 'min') else r),
            )
        s = add_p_stars(s, p)           # TODO: add pad if we add a mean column?
        s = highlight_extreme(
            s, props='textbf:--latex--rwrap;', op=best
            )  # Inside p_stars
        col_fmt = 'l' * s.data.index.nlevels + ('c') * len(s.data.columns)
        tex = kill_extra_cline(s.to_latex(
            hrules=True,
            clines='skip-last;data',
            column_format=col_fmt,
            convert_css=True  # for background_gradient
            ))
        latex_id = results_id.replace('-', '_')
        with open(f'experiments/tables/{latex_id}_corr.tex', 'w') as fo:
            fo.write(tex)

        for stats_id, index_levels in (
            ('stats-corpora', 2),
            ('stats-datasets', 1),
            ):
            csv_index = list(range(index_levels))
            stats = pd.read_csv(
                f'experiments/{stats_id}.csv',
                index_col=csv_index
                )  # TODO doesn't need limit_to_valid_corpora

            if stats_id == 'stats-datasets':
                lang_note = pd.Series(stats.columns).str.partition(' ')
                # Only if not empty:
                if (lang_note[2] != '').any():
                    stats.columns = pd.MultiIndex.from_arrays(
                        [lang_note[0], lang_note[2]], names=None
                        )

            s = Styler(stats, precision=STAT_PREC, thousands=',', na_rep='---')

            if stats_id == 'stats-corpora':
                s = s.apply(alternate_row_bg, axis=None)
                s = s.apply_index(alternate_row_bg, level=(index_levels - 1))
            # TODO? use siunitx to add thousands separators
            col_fmt = 'l' * s.data.index.nlevels + ('r') * len(s.data.columns)
            tex = s.to_latex(
                hrules=True,
                # TODO add groups too? clines='skip-last;data',
                # + kill_extra_cline()
                column_format=col_fmt,
                sparse_columns=False,   # repeat language names for datasets
                #  convert_css=True # for background_gradient
                )
            latex_id = stats_id.replace('-', '_')
            with open(f'experiments/tables/{latex_id}.tex', 'w') as fo:
                fo.write(tex)


if __name__ == '__main__':
    main()
