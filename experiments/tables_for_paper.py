from functools import partial
import pandas as pd
import numpy as np
from pandas.io.formats.style import Styler  # type: ignore
from pandas.io.formats.style_render import Subset

RESULT_PREC = 3
RESULT_TOL = 0.0001  # We use at most 3 decimal places, so it's appropriate

VALID_CORPORA = {
    'Wikipedia': 'wiki',
    'wordfreq': 'wordfreq',
    'SUBTLEX': 'subtlex',
    'SubIMDB': 'subimdb',
    'OpenSubtitles': 'os',
    'GINI': 'gini',
    'CSJ': 'csj-lemma',
    'TUBELEX': 'tubelex',
    'TUBELEX\\textsubscript{R}': 'tubelex-regex',
    'TUBELEX\\textsubscript{B}': 'tubelex-base',
    'TUBELEX\\textsubscript{L}': 'tubelex-lemma',
    }



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
    axis=0, # by column
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
    valid = [c in VALID_CORPORA for c in df.index]
    return df.loc[valid]

def main():
    for results_id, best in (
        ('mlsp', 'min'),
        ('ldt', 'min'),
        ('fam', 'max')
        ):
        r = limit_to_valid_corpora(pd.read_table(
            f'experiments/{results_id}-corr-aggregate-correlation.tsv',
            index_col=0
            ))
        p = limit_to_valid_corpora(pd.read_table(
            f'experiments/{results_id}-corr-aggregate-pvalues.tsv',
            index_col=0
            ))


        s = Styler(r, precision=RESULT_PREC, na_rep='---')

        r_not_na = ~r.isna()
        s = background_gradient(
            s,
            cmap='Blues', bool_subset=r_not_na,
            gmap=(-r if (best == 'min') else r),
            )
        s = add_p_stars(s, p)           # TODO: add pad if we add a mean column?
        s = highlight_extreme(s, props='textbf:--latex--rwrap;', op=best)  # Inside p_stars
        col_fmt = 'l' * s.data.index.nlevels + ('c') * len(s.data.columns)
        tex = s.to_latex(
            hrules=True,
            column_format=col_fmt,
            convert_css=True # for background_gradient
            )
        with open(f'experiments/tables/{results_id}_corr.tex', 'w') as fo:
            fo.write(tex)


if __name__ == '__main__':
    main()
