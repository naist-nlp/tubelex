import argparse
import pandas as pd
import os
import re
from tqdm import tqdm
from tubelex import get_write_file
from freq_utils import Storage

CATEGORIES = {
    'alltyp1': 'Spoken demographic',
    'scgdom1': 'Educational/Informative',
    'scgdom2': 'Business',
    'scgdom3': 'Public/Institutional',
    'scgdom4': 'Leisure',
    'wridom1': 'Imaginative',
    'wridom2': 'Informative: natural & pure science',
    'wridom3': 'Informative: applied science',
    'wridom4': 'Informative: social science',
    'wridom5': 'Informative: world affairs',
    'wridom6': 'Informative: commerce & finance',
    'wridom7': 'Informative: arts',
    'wridom8': 'Informative: belief & thought',
    'wridom9': 'Informative: leisure'
    }


CAT_PAT = re.compile('<catRef targets="([^"]+)"/>')


def get_cat(fields: list[str]):
    c = [f for f in fields if f in CATEGORIES]
    assert len(c) == 1   # each file should belong to a single category
    return c[0]

def traverse_paths_names(directory: str):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.xml'):
                file_path = os.path.join(root, file)
                name = file.removesuffix('.xml')
                yield (file_path, name)

def main(args):
    storage = Storage.from_args(args)

    if args.texts:
        ids = []
        cats = []
        paths_ids = sorted(list(traverse_paths_names(args.texts)))
        for path, id in paths_ids:
            with open(path) as f:
                first_line = f.readline()
                m = CAT_PAT.search(first_line)
                assert m is not None
                c = get_cat(m.group(1).lower().split(' '))
                ids.append(id)
                cats.append(c)

        if args.write_corpus:
            prefix = args.texts + '/'
            from nltk.corpus.reader.bnc import BNCCorpusReader
            with get_write_file(args.corpus_path, storage) as write_file:
                for path, id in tqdm(paths_ids):
                    fileids = [path.removeprefix(prefix)]
                    words = BNCCorpusReader(root=args.texts, fileids=fileids).words()
                    write_file(f'{id}.txt', ' '.join(words))

    else:
        with open(args.input_file) as f:
            entries = [line.rstrip().split(' ') for line in f]

        ids = [e[0] for e in entries]
        cats = [get_cat(e[1:]) for e in entries]

    df = pd.DataFrame({
        'videoid': ids,
        'categories': cats
        })

    df.to_csv(args.output_file, index=False)

    cat_counts = df['categories'].value_counts()

    print(r'Classification & \#Texts & Register & Description\\')
    print(r'\midrule')
    for cat, desc in CATEGORIES.items():
        n = cat_counts[cat]
        r = 'Written' if cat.startswith('wri') else 'Spoken'
        desc = desc.replace('&', r'\&')
        print(rf'{cat} & {n} & {r} & {desc}\\')
    print(r'\midrule')
    print(rf'Total & {cat_counts.sum()} &\\')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--texts')  # the Texts directory (will ignore --input-file)

    parser.add_argument('--input-file', default='data/BNC/BNC/disk2/SGML/bncfinder.dat')

    parser.add_argument('--output-file', '-o', default='data/bnc.csv')

    parser.add_argument('--write-corpus', action='store_true')
    parser.add_argument('--corpus-path', default='data/bnc')
    Storage.add_arg_group(parser, 'Compression options', zip_suffix=True)

    args = parser.parse_args()
    main(args)
