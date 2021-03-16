import pandas as pd


def correlation_var(df, var1, var2):
    correlation = df[var1].corr(df[var2])
    print("Correlation between " + var1 + " and" + var2 + " is:", correlation)


if __name__ == "__main__":
    df_2014 = pd.read_csv('a.csv')
    df_2015 = pd.read_csv('b.csv')
    # Concatenate datasets
    df = pd.concat([df_2014, df_2015], sort=False)
    correlation_var(df, 'reenrolled', 'bsa_dummy')
