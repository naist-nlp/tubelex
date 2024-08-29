import argparse
from freq_utils import Storage
from tubelex import (
    UNIQUE_PATH_FMT,
    get_files_contents,
    )
from collections import Counter
from typing import Optional
import sys
from tqdm import tqdm
import hanzidentifier as hzid

'''
Usage (for LZMA compression):

python analyze_zh.py -z
'''

HANZI_ID2DESC = {
    hzid.UNKNOWN: 'Unknown',
    hzid.BOTH: 'Both',
    hzid.TRADITIONAL: 'Traditional',
    hzid.SIMPLIFIED: 'Simplified',
    hzid.MIXED: 'Mixed',
    }


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(
        'Analyze Chinese variants'
        ))
    Storage.add_arg_group(parser, 'Compression options', zip_suffix=True)
    parser.add_argument('--strict', '-s', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Log info about non-strict identification to stdout.')
    return parser.parse_args()


def get_example(lines: list[str], kind: int) -> Optional[str]:
    if (s := next((s for s in lines if hzid.identify(s) == kind), None)):
        if kind == hzid.MIXED:
            tcs = ''.join(c for c in s if hzid.identify(c) == hzid.TRADITIONAL)
            scs = ''.join(c for c in s if hzid.identify(c) == hzid.SIMPLIFIED)
            return f'T={tcs}, S={scs} in {s}'
        cs = ''.join(c for c in s if hzid.identify(c) == kind)
        return f'{cs} in {s}'
    return None


def identify_majority(
        s: str,
        name: Optional[str] = None,  # log if MIXED
        min_prop: float = 0.95,
        max_ignore: int = 1,
        ) -> int:
    '''
    Return the majority category (by counting lines).
    Can be one of Traditional/Simplified, if the proportion (including lines that can
    be both) is >= min_prop OR the number of lines that belongs to the other category
    (not including lines that can be both) is <= max_ignore.
    '''
    lines = s.splitlines()
    c = Counter(map(hzid.identify, lines))
    n = c.total() - c[hzid.UNKNOWN]
    if not n:
        return hzid.UNKNOWN
    both = c[hzid.BOTH]
    if both == n:
        return hzid.BOTH
    trad = c[hzid.TRADITIONAL]
    simp = c[hzid.SIMPLIFIED]
    trad_prop = (trad + both) / n
    simp_prop = (simp + both) / n
    if (trad_prop > simp_prop) and ((trad_prop >= min_prop) or (simp <= max_ignore)):
        return hzid.TRADITIONAL
    if (simp_prop > trad_prop) and ((simp_prop >= min_prop) or (trad <= max_ignore)):
        return hzid.SIMPLIFIED

    if name is not None:
        ext = get_example(lines, hzid.TRADITIONAL)
        exs = get_example(lines, hzid.SIMPLIFIED)
        exm = get_example(lines, hzid.MIXED)
        print(
            f'{name}: {n} lines = {trad} Trad. + {simp} Simp. + '
            f'{both} both + {c[hzid.MIXED]} Mixed, e.g.:\n'
            f'- T: {ext}\n'
            f'- S: {exs}\n'
            f'- M: {exm}\n'
            )

    return hzid.MIXED


def main() -> None:
    args = parse()
    storage = Storage.from_args(args)

    identify = hzid.identify if args.strict else identify_majority
    with get_files_contents(
        UNIQUE_PATH_FMT % 'zh', storage,
        ) as files_contents:
        files, iter_contents = files_contents
        id_iter = (
            map(identify, iter_contents(), files) if (args.verbose and not args.strict)
            else map(identify, iter_contents())
            )
        zh_id_counts = Counter(tqdm(
            desc='Identifiying Chinese variant in unique files',
            iterable=id_iter,
            total=len(files)
            ))
    with open('corpus/variants-zh.csv', 'w') as f:
        print('variant,count', file=f)
        for hanzi_id, n in zh_id_counts.items():
            print(f'{HANZI_ID2DESC[hanzi_id]},{n}', file=f)


if __name__ == '__main__':
    main()
