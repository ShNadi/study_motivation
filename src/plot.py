import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_var(df, type='boxplot', Y='age', X='bsa_dummy', HUE= None):
    if type == 'boxplot':
        bplot = sns.boxplot(y=Y, x=X, hue=HUE,
                            data=df,
                            width=0.5, showmeans=True)
        bplot.axes.set_title(Y + ' Vs ' + X + ":\n", fontsize=14)

        bplot.set_xlabel(X, fontsize=12)

        bplot.set_ylabel(Y, fontsize=12)

        bplot.tick_params(labelsize=8)
        plt.show()

    if type == 'correlation_matrix':
        f = plt.figure(figsize=(8, 10))
        plt.matshow(df.corr(), fignum=f.number)
        plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=10,
                   rotation=90)
        plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=10)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title('Correlation Matrix', fontsize=16,  y=-0.1);
        plt.show()

    if type == 'correlation_matrix_numbers':
        fig, ax = plt.subplots(figsize=(15, 15))
        correlation_mat = df.corr()
        sns.heatmap(correlation_mat, annot=True)
        plt.show()


if __name__ == "__main__":
    # df = pd.read_csv('../data/processed/motivation.csv')
    # plot_var(df, type='boxplot', Y='age', X='bsa_dummy', HUE='year')

    # df = pd.read_csv('../data/processed/motivation_liwic_meta_pos_topic.csv')
    # plot_var(df, type='boxplot', Y='nr_token', X='bsa_dummy')
    # plot_var(df, type='boxplot', Y='nr_adj', X='bsa_dummy')
    # plot_var(df, type='boxplot', Y='nr_noun', X='bsa_dummy')
    # plot_var(df, type='boxplot', Y='nr_verb', X='bsa_dummy')
    # plot_var(df, type='boxplot', Y='nr_number', X='bsa_dummy')

    # plot_var(df, type='correlation_matrix')

    # df = pd.read_csv('../data/processed/motivation_liwc_meta_pos.csv')
    # plot_var(df, type='correlation_matrix')

    df = pd.read_csv('../data/processed/motivation_liwc_meta_pos.csv')
    plot_var(df, type='correlation_matrix_numbers')







