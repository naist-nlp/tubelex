import argparse
import pandas as pd
from gensim.models import KeyedVectors
from frequency_data import download_if_necessary
ANALOGY_ID2URL = {
    # Google/Mikolov (mikolov_etal_2013_efficient)
    'en': 'https://github.com/tmikolov/word2vec/raw/master/questions-words.txt',
    # Spanish translation - https://crscardellino.net/SBWCE/
    'es': 'https://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/questions-words_sp.txt',
    # Chinese small subset translation via chen_etal_2015_joint:
    'zh': ('https://github.com/Embedding/Chinese-Word-Vectors/raw/master/'
           'testsets/CA_translated/ca_translated.txt'),
    # Chinese (original)
    'zh-morph': ('https://github.com/Embedding/Chinese-Word-Vectors/blob/master/'
                 'testsets/CA8/morphological.txt'),
    'zh-sem': ('https://github.com/Embedding/Chinese-Word-Vectors/blob/master/'
               'testsets/CA8/semantic.txt')
    }
SIMILARITY_ID2URL = {
    'en': 'https://multisimlex.com/data/ENG.csv',
    'es': 'https://multisimlex.com/data/SPA.csv',
    'zh': 'https://multisimlex.com/data/CMN.csv'
    }


def analogy_path(data_id):
    return f'data/downloads/analogy-{data_id}.txt'


def similarity_path(data_id, mean=False):
    return (
        f'data/downloads/similarity-mean-{data_id}.txt' if mean else
        f'data/downloads/similarity-{data_id}.txt'
        )

# FASTTEXT_DIR    = 'data/downloads/fasttext'
#

# def fasttext_path(lang: str, dir: str) -> str:
#     return os.path.join(dir, f'cc.{lang}.300.bin')
#
# def download_fasttext(lang: str, dir: str = FASTTEXT_DIR) -> str:
#     path = fasttext_path(lang, dir)
#     if not os.path.exists(path):
#         _mkdir_parent(path)
#
#         old_wd = os.getcwd()
#         with TemporaryDirectory(dir=old_wd) as tmp_wd:
#             # Using temporary directory to:
#             # (1) automatically cleanup .gz (successful download)
#             #     or .gz.part (unsuccessful donwload) left by download_model(),
#             # (2) avoid conflicts with any parallel process.
#             os.chdir(tmp_wd)
#             with redirect_stdout(sys.stderr):
#                 dl_path = download_model(lang, if_exists='ignore')  # calls print()
#             os.rename(dl_path, os.path.join(old_wd, path))
#             os.chdir(old_wd)
#         sys.stderr.write(f'Moved FastText download to "{path}".\n')
#
#     return path
#

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--similarity', action='store_true',
                        help='Evaluate similarity instead of analogy.')
    parser.add_argument('--restrict-vocab', type=int, default=300000,
                        help='Top n.')
    parser.add_argument('language', #default='en',
                        help='Language (or id such as zh-morph).')
    parser.add_argument('model', help='Pathed to the *.vec file.')

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    for data_id, url in ANALOGY_ID2URL.items():
        download_if_necessary(url, analogy_path(data_id))
    for data_id, url in SIMILARITY_ID2URL.items():
        path_raw = similarity_path(data_id)
        path_mean = similarity_path(data_id, mean=True)
        download_if_necessary(url, path_raw)
        df = pd.read_csv(path_raw)
        df_mean = df[['Word 1','Word 2']].copy()
        df_mean['Similarity'] =  df[
            [c for c in df.columns if c.startswith('Annotator ')]
            ].mean(axis=1) / 6      # 0 to 6
        df_mean.to_csv(path_mean, header=False, index=False, sep='\t')

    # Load FastText word embeddings (pre-trained model)
    data_id = args.language
    # lang    = data_id if ('-' not in data_id) else data_id.split('-', 1)[0]

    kv = KeyedVectors.load_word2vec_format(args.model)

    if args.similarity:
        (
            (pearson, pearson_p),
            (spearman, spearman_p),
            oov_ratio
            ) = kv.evaluate_word_pairs(
                similarity_path(data_id, mean=True),
                dummy4unknown=True, restrict_vocab=args.restrict_vocab
                )
        oov_ratio /= 100  # percentage, duh
        print('pearson\t%.3f' % pearson)
        print('pearson_p\t%.3f' % pearson_p)
        print('spearman\t%.3f' % spearman)
        print('spearman_p\t%.3f' % spearman_p)
        print('oov_ratio\t%.3f' % oov_ratio)
        return


    acc, results = kv.evaluate_word_analogies(
        analogy_path(data_id),
        dummy4unknown=True, restrict_vocab=args.restrict_vocab
        )

    total_correct = 0
    total_qs = 0

    total_se_geo_correct = 0
    total_se_geo_qs = 0

    total_se_correct = 0
    total_se_qs = 0

    total_sy_correct = 0
    total_sy_qs = 0

    for r in results:
        name = r['section']
        nc = len(r['correct'])
        nq = nc + len(r['incorrect'])

        total_correct               += nc
        total_qs                    += nq
        if name == 'family':
            total_se_correct        += nc
            total_se_qs             += nq
        elif name.startswith('gram'):
            total_sy_correct        += nc
            total_sy_qs             += nq
        elif name == 'Total accuracy':
            continue    # ignore
        else:
            assert name in {'capital-common-countries', 'capital-world', 'currency',
                            'city-in-state'}, name  # geographical
            total_se_correct        += nc
            total_se_qs             += nq
            total_se_geo_correct    += nc
            total_se_geo_qs         += nq

    acc_all = total_correct / total_qs
    acc_sem = total_se_correct / total_se_qs
    acc_syn = total_sy_correct / total_sy_qs
    acc_geo = total_se_geo_correct / total_se_geo_qs
    acc_fam = ((total_se_correct - total_se_geo_correct) /
               (total_se_qs - total_se_geo_qs))

    print('geography\t%.3f' % acc_geo)
    print('family\t%.3f' % acc_fam)
    print('semantic\t%.3f' % acc_sem)
    print('syntactic\t%.3f' % acc_syn)
    print('all\t%.3f' % acc_all)

    assert abs(acc - acc_all) < 0.0001


if __name__ == '__main__':
    main(parse_args())
