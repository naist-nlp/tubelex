import argparse
from os import listdir
import os
import re
import sys
from csv import QUOTE_NONE
from collections import defaultdict
import pandas as pd

HEADING_IDS     = '* video IDs in the list:\n'
KEY_IDS         = 'ids'

KEY2DESC_RE = {
    # ids are listed like this after `HEADING_IDS`:
    # * video IDs in the list:
    #   120000
    KEY_IDS: r'',
    # We do not need to check the following headings:
    # * files:
    'files': r'total',
    'files_short': r'too short .*',
    'files_little_chars': r'not enough [a-z]+ characters .*',
    'files_little_lang': r'not enough detected .*',
    'files_valid': r'valid files after cleaning',
    # * sequences removed from valid files:
    'rm_tags': r'tags',
    'rm_addresses': r'addresses',
    # * lines in valid files:
    'lines': r'total lines',
    'lines_ws': r'whitespace-only lines',
    'lines_no_chars': r'lines composed of non-[a-z]+ characters',
    'lines_valid': r'valid lines after cleaning',
    # * VTT cue cleanup:
    'cues': r'total cues',
    'cues_empty': r'empty cues',
    'cues_repeated': r'repeated cues',
    'cues_scrolling': r'scrolling cues',
    # ignore the 2nd 'total' (in the "duplicate" section  == 'files_valid' above)
    'dedup_removed': r'duplicates removed',
    'dedup_valid': r'valid files',
    'cc_descriptions': r'CC descriptions filtered'
    }

VALUE_DESC_RE = r'  ?(?P<value>[0-9]+)( (?P<desc>.+))?\n'


def tubelex_out2dict(
    path: str, d: dict[str, int] | None = None
    ) -> dict[str, int]:
    if d is None:
        d = {}
    with open(path) as f:
        after_heading_ids: bool = False
        for line in f:
            if (m := re.fullmatch(VALUE_DESC_RE, line)):
                value = int(m.group('value'))
                desc = m.group('desc') or ''    # may be None -> ''
                for key, desc_re in KEY2DESC_RE.items():
                    if key == KEY_IDS and not after_heading_ids:
                        continue
                    if re.fullmatch(desc_re, desc):
                        if key not in d:
                            d[key] = value
                        break
                after_heading_ids = False
            else:
                after_heading_ids = (line == HEADING_IDS)
    if (missing := set(KEY2DESC_RE).difference(d)):
        raise Exception(f'Items {missing} are missing in TUBELEX output {path}')
    return d


TUBELEX_OUT_RE = r'tubelex-(?P<lang>[a-z]+)\.out'
TUBELEX_FREQ_RE = r'tubelex-(?P<lang>[a-z]+)(-(?P<id>[a-z-]+))?-nfkc-lower\.tsv\.xz'
TOTAL_ROW = '[TOTAL]'


def tubelex_freq2dict(
    path: str,
    freq_id: str,
    d: dict[str, int] | None = None
    ) -> dict[str, int]:
    if d is None:
        d = {}
    df = pd.read_table(
        path, quoting=QUOTE_NONE, usecols=['word', 'count'], index_col='word'
        )
    c = df['count']
    total = c[TOTAL_ROW]
    c.drop(TOTAL_ROW, inplace=True)
    n_tokens = c.sum()
    assert n_tokens == total
    d[f'{freq_id}:types']   = len(c)
    d[f'{freq_id}:tokens']  = n_tokens
    return d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--frequencies', default='frequencies',
                        help='Tubelex frequencies directory (input)')
    action.add_argument('--output', '-o', default='experiments/stats.csv',
                        help='Stats output file (CSV)')
    return parser.parse_args()


def main(args: argparse.Namespace):
    path    = args.frequencies

    lang2d = defaultdict(dict)

    for name in sorted(
        # Process .out files first => nicer column ordering:
        listdir(path), key=(lambda name: not name.endswith('.out'))
        ):

        p = os.path.join(path, name)
        if not os.path.isfile(p):
            continue

        if (m := re.fullmatch(TUBELEX_OUT_RE, name)) is not None:
            tubelex_out2dict(p, lang2d[m.group('lang')])
        elif (m := re.fullmatch(TUBELEX_FREQ_RE, name)) is not None:
            print(f'Processing frequencies: {name}...', file=sys.stderr)
            freq_id = (m.group('id') or 'default').replace('-', '_')
            tubelex_freq2dict(p, freq_id, lang2d[m.group('lang')])

    df = pd.DataFrame(lang2d).transpose()       # rows = languages
    df.index.name = 'lang'
    df.to_csv(args.output, float_format='%d')   # prevent .0 floats
    print('Done.', file=sys.stderr)


if __name__ == '__main__':
    args = parse_args()
    main(args)
