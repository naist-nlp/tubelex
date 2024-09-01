import re
from itertools import groupby
from collections.abc import Iterator, Iterable
import pandas as pd


def escape_bs(s: str) -> str:
    return s.replace('\\', r'\\')


def lens_to_latex_ranges(
    lens: Iterable[int], start: int = 1
    ) -> Iterator[tuple[int, int]]:
    i = start
    for l in lens:
        j = i + l - 1
        yield (i, j)
        i = j + 1


def top_level_lens(idx: pd.MultiIndex) -> Iterator[int]:
    return map(lambda g_it: len(list(g_it[1])), groupby(idx.codes[0]))


def top_level_lines(df: pd.DataFrame, cmd: str = 'cline') -> str:
    return ' '.join(map(
        lambda ij: rf'\{cmd}{{{ij[0]}-{ij[1]}}}',
        lens_to_latex_ranges(
            top_level_lens(df.columns),
            start=df.index.nlevels + 1
            )
        ))


def colapse_latex_table_header(latex: str, df: pd.DataFrame) -> str:
    column_nlevels = df.columns.nlevels

    re_header = r'(?<=\\toprule\n)(.*)(?=\\midrule\n)'
    header = re.search(
        re_header, latex, flags=re.DOTALL
        ).group(1)
    header_rows = header.split('\\\\\n')
    assert header_rows[-1] == ''
    header_nrows = len(header_rows) - 1
    assert column_nlevels <= header_nrows <= column_nlevels + 1
    if header_nrows > column_nlevels:
        index_nlevels = df.index.nlevels
        re_prefix = rf'^([^&]*&){{{index_nlevels}}}'
        index_header_prefix = re.match(
            re_prefix, header_rows[-2]
            ).group()
        header_rows[-3] = re.sub(
            re_prefix, escape_bs(index_header_prefix), header_rows[-3]
            )
        del header_rows[-2]

    # if column_nlevels > 1:
    #     # Add header lines
    #     top_lines = top_level_lines(df, cmd='cmidrule(l{1pt}r{1pt})')
    #     header_rows[1] = f'{top_lines}\n{header_rows[1]}'

    header = '\\\\\n'.join(header_rows)
    return re.sub(re_header, escape_bs(header), latex, flags=re.DOTALL)
