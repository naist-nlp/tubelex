import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text


def main():
    for task in ('mlsp', 'ldt', 'fam'):    # no fam
        df = pd.read_csv(f'experiments/stats-size-correlation-{task}.csv', index_col=0)
        fig_path = f'experiments/figures/{task}_size_correlation.pdf'

        df['log10_tokens'] = np.log10(df['tokens'])

        corpus_kind = list(pd.Series(df.index).str.partition(':')[0])

        plt.figure(figsize=(5, 3))
        if task != 'fam':
            plt.gca().invert_yaxis()
        sns.scatterplot(
            x=df['log10_tokens'], y=df['correlation'], hue=corpus_kind, legend=False
            )

        texts = []
        for i in range(df.shape[0]):
            text = df.index[i]
            texts.append(
                plt.text(df.loc[text, 'log10_tokens'], df.loc[text, 'correlation'],
                         text,
                         ha='center',
                         fontweight=('bold' if text.startswith('TUBELEX') else 'normal')
                         )
                )

        adjust_text(
            texts,
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
            min_arrow_len=0,
            )

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
