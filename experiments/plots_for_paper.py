import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text


def abbr_corpus(corp_lang: str) -> str:
    corp, lang = corp_lang.split(':')
    corp = 'SUBT-UK' if (corp == 'SUBTLEX-UK') else corp[:4].replace('-', '')
    return ':'.join((corp, lang))


def main():
    for task in ('mlsp', 'ldt', 'fam'):
        df = pd.read_csv(f'experiments/stats-size-correlation-{task}.csv', index_col=0)
        fig_path = f'experiments/figures/{task}_size_correlation.pdf'

        df = df.loc[
            ~df.index.str.startswith('TUBELEX') |
            df.index.str.contains('default')
            ]

        df['log10_tokens'] = np.log10(df['tokens'])

        corpus_kind = list(
            pd.Series(df.index).str.partition(':')[0].str.partition('-')[0]
            )

        fig = plt.figure(figsize=(4.5, 2.7))
        if task != 'fam':
            plt.gca().invert_yaxis()
        sns.scatterplot(
            x=df['log10_tokens'], y=df['correlation'], hue=corpus_kind, legend=False,
            s=60    # slightly larger size (default is <40?)
            )

        texts = []
        for i in range(df.shape[0]):
            idx = df.index[i]
            text = abbr_corpus(idx)
            texts.append(
                plt.text(df.loc[idx, 'log10_tokens'], df.loc[idx, 'correlation'],
                         text,
                         ha='center',
                         fontweight=('bold' if idx.startswith('TUBELEX') else 'normal')
                         )
                )
        texts, patches = adjust_text(
            texts,
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.15),
            # Doesn't work SMH min_arrow_len=30,
            )

        # Manual fixes:
        if task == 'fam':
            for text in texts: #text, patch in zip(texts, patches):
                if text._text == 'SubI:en':
                    print('Fixing SubI:en')
                    # patch.remove()
                    text._x = text._x - 0.35
                    text._y = text._y - 0.035
                elif text._text == 'TUBE:en':
                    print('Fixing TUBE:en')
                    text._x = text._x - 0.8
                elif text._text == 'SUBT:en':
                    print('Fixing SUBT:en')
                    text._x = text._x
                    text._y = text._y - 0.035
                elif text._text == 'SUBT-UK:en':
                    print('Fixing SUBT-UK:en')
                    text._y = text._y - 0.005
        elif task == 'mlsp':
            plt.yticks([-0.5, -0.6, -0.7])

        fig.canvas.draw_idle()

        plt.xlabel('Corpus Size: log$_{10}$(#tokens)')
        plt.ylabel(
            'Lexical Complexity Correlation' if task == 'mlsp' else
            'Word Familiarty Correlation' if task == 'fam' else
            'LDT Correlation'
            )
        # plt.title(f'{task}: Coverage vs. corpus size')

        plt.savefig(fig_path, bbox_inches='tight')


if __name__ == '__main__':
    main()
