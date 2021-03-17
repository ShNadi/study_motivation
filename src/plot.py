import pandas as pd
import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt


def plot_var(df, type='boxplot', Y='age', X='bsa_dummy', HUE= None):
    if type == 'boxplot':
        bplot = sns.boxplot(y=Y, x=X, hue=HUE,
                            data=df,
                            width=0.5)
        bplot.axes.set_title(Y + ' Vs ' + X + ":\n", fontsize=14)

        bplot.set_xlabel(X,
                         fontsize=12)

        bplot.set_ylabel(Y,
                         fontsize=12)

        bplot.tick_params(labelsize=8)
        plt.show()



if __name__ == "__main__":
    df = load_dataset.read_df()
    plot_var(df, type='boxplot', Y='age', X='bsa_dummy', HUE='year')