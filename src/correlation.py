import pandas as pd
import load_dataset

#There're two target variables 'reenrolled', 'bsa_dummy'
# Find correlation between target variables on concatenated dataset(df_2014 and df_2015)

def correlation_var(df, var1, var2):
    correlation = df[var1].corr(df[var2])
    print("Correlation between " + var1 + " and" + var2 + " is:", correlation)


if __name__ == "__main__":
    df = load_dataset.read_df()
    correlation_var(df, 'reenrolled', 'bsa_dummy')
