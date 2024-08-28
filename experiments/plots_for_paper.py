import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text



def main():
    # Sample DataFrame with index as labels
    for task in ('mlsp', 'ldt'):    # no fam
        data_path = f'experiments/stats-coverage-{task}.csv'
        fig_path = data_path.replace('.csv', '.pdf')
        df = pd.read_csv(f'experiments/stats-coverage-{task}.csv', index_col=0)

        df['log10_tokens'] = np.log10(df['tokens'])

        # Create a scatterplot

        corpus_kind = list(pd.Series(df.index).str.partition(':')[0])

        #sns.set(rc={"figure.figsize":(6, 8)}) #width=3, #height=4
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=df['log10_tokens'], y=df['coverage'], hue=corpus_kind, legend=False
            )

        # Annotate doesn't work with adjust_text:
        #     for i in range(df.shape[0]):
        #         plt.annotate(df.index[i], (df['log10_tokens'][i], df['coverage'][i]),
        #                      textcoords="offset points", xytext=(5,-5), ha='center')

        texts = []
        for i in range(df.shape[0]):
            texts.append(
            #             plt.annotate(df.index[i], (df['log10_tokens'][i], df['coverage'][i]),
            #                          textcoords="offset points", xytext=(0,0), ha='center')
                plt.text(df['log10_tokens'][i], df['coverage'][i], df.index[i],
                             ha='center')
            )

        # np.random.seed(seed)
        adjust_text(
            texts,
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
            min_arrow_len=0,
            force_explode=(1,1),
            force_pull=(0,0)
            ) # ensure the labeling is clear by adding arrows)

        # Adding labels and title
        plt.xlabel('Corpus size: log10(#tokens)')
        plt.ylabel('Data coverage')
        # plt.title(f'{task}: Coverage vs. corpus size')

        plt.savefig(fig_path)

if __name__ == '__main__':
    main()
