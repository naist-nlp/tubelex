import argparse
from freq_utils import Storage
from tubelex import (
    UNIQUE_PATH_FMT,
    get_files_contents,
    )
import hanzidentifier
from collections import Counter
from tqdm import tqdm

'''
Usage (for LZMA compression):

python analyze_zh.py -z
'''

HANZI_ID2DESC = {
    hanzidentifier.UNKNOWN: 'Unknown',
    hanzidentifier.BOTH: 'Both',
    hanzidentifier.TRADITIONAL: 'Traditional',
    hanzidentifier.SIMPLIFIED: 'Simplified',
    hanzidentifier.MIXED: 'Mixed',
    }


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(
        'Analyze Chinese variants'
        ))
    Storage.add_arg_group(parser, 'Compression options', zip_suffix=True)
    parser.add_argument('--strict', action='store_true')
    return parser.parse_args()


def identify_majority(s: str, min_prop: float = 0.95, max_ignore: int = 1) -> int:
    '''
    Return the majority category (by counting lines).
    Can be one of Traditional/Simplified, if the proportion (including lines that can
    be both) is >= min_prop OR the number of lines that belongs to the other category
    (not including lines that can be both) is <= max_ignore.
    '''
    c = Counter(map(hanzidentifier.identify, s.splitlines()))
    n = c.total() - c[hanzidentifier.UNKNOWN]
    if not n:
        return hanzidentifier.UNKNOWN
    both = c[hanzidentifier.BOTH]
    if both == n:
        return hanzidentifier.BOTH
    trad = c[hanzidentifier.TRADITIONAL]
    simp = c[hanzidentifier.SIMPLIFIED]
    trad_prop = (trad + both) / n
    simp_prop = (simp + both) / n
    if (trad_prop > simp_prop) and ((trad_prop >= min_prop) or (simp <= max_ignore)):
        return hanzidentifier.TRADITIONAL
    if (simp_prop > trad_prop) and ((simp_prop >= min_prop) or (trad <= max_ignore)):
        return hanzidentifier.SIMPLIFIED
    return hanzidentifier.MIXED


def main() -> None:
    args = parse()
    storage = Storage.from_args(args)

    identify = hanzidentifier.identify if args.strict else identify_majority
    with get_files_contents(
        UNIQUE_PATH_FMT % 'zh', storage,
        ) as files_contents:
        files, iter_contents = files_contents
        zh_id_counts = Counter(tqdm(
            desc='Identifiying Chinese variant in unique files',
            iterable=map(identify, iter_contents()),
            total=len(files)
            ))
    with open('corpus/variants-zh.csv', 'w') as f:
        print('variant,count', file=f)
        for hanzi_id, n in zh_id_counts.items():
            print(f'{HANZI_ID2DESC[hanzi_id]},{n}', file=f)


if __name__ == '__main__':
    main()
