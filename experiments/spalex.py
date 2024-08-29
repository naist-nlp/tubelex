import os
import sys
import pandas as pd
from frequency_data import download_if_necessary

SPALEX_DIR = 'data/downloads/spalex/'

# SPALEX_WORD_INFO = SPALEX_DIR + 'word_info.csv'
SPALEX_LEXICAL = SPALEX_DIR + 'lexical.csv'
SPALEX_MEAN = SPALEX_DIR + 'mean.csv'

SPALEX_PATH2URL = {
    # SPALEX_WORD_INFO: 'https://figshare.com/ndownloader/files/11826623',
    SPALEX_LEXICAL: 'https://figshare.com/ndownloader/files/11209613'
    }


def get_spalex(redo=False):
    if redo or not os.path.exists(SPALEX_MEAN):
        for path, url in SPALEX_PATH2URL.items():
            download_if_necessary(url, path)

        lexical = pd.read_csv(
            # NA values from SPALEX's merge.R, but there are no NA values actually.
            SPALEX_LEXICAL, na_values=['', 'NA'], keep_default_na=False
            )

        # Compute mean RT for words (exclude non-words) including incorrect trials:
        print('Computing mean RT', file=sys.stderr)
        is_w = (lexical['lexicality'] == 'W')
        lexical = lexical.loc[is_w, ['spelling', 'rt']]

        # "Trim" RT/ms to [200, 2000] based on (Aguasvivas et al., 2018)
        # Trimming could be:
        #  lexical.loc[lexical['rt'] < 200, 'rt'] = 200
        #  lexical.loc[lexical['rt'] > 2000, 'rt'] = 2000
        # I assume it's removal:
        lexical.drop(lexical.index[lexical['rt'] < 200], inplace=True)
        lexical.drop(lexical.index[lexical['rt'] > 2000], inplace=True)

        # Not doing this, since the explanation is unclear:
        # "removing outliers above and below 1.5 box lengths" (Aguasvivas et al., 2018)
        # box length is a quartile (0.25), assuming from median (0.5)
        # Probably supposed to be done on a per-participant basis?
        #
        # lo = np.quantile(lexical['rt'], 0.5 - 0.25 * 1.5)
        # lexical.drop(lexical.index[lexical['rt'] < lo], inplace=True)
        #
        # hi = np.quantile(lexical['rt'], 0.5 + 0.25 * 1.5)
        # lexical.drop(lexical.index[lexical['rt'] > hi], inplace=True)

        mean_rt = lexical.groupby('spelling').mean()
        print(
            f'Overall mean for words = {mean_rt.mean().item()} ms, '
            f'sd = {mean_rt.std().item()} ms',
            file=sys.stderr
            )

        mean_rt.rename_axis(index='word', inplace=True)  # 'spelling' -> 'word'

        mean_rt.to_csv(SPALEX_MEAN)

    return pd.read_csv(SPALEX_MEAN)


if __name__ == '__main__':
    assert len(sys.argv) <= 2
    get_spalex(redo=((len(sys.argv) == 2) and (sys.argv[1] in {'--redo', '-r', '-R'})))
