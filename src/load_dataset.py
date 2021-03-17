import pandas as pd

def read_df( status = 'read'):
    if status == 'prepare':
        # Read df_2014 and df_2015
        columns = ['studentnr_crypt', 'cohort', 'program', 'field', 'motivation',
                        'prior_educ', 'previously_enrolled', 'multiple_requests', 'age',
                        'gender', 'HSGPA', 'interest', 'ase', 'reenrolled', 'bsa_dummy']

        df_2014 = pd.read_csv(r'..\data\raw\motivatoin_2014.csv')
        df_2014 ['year'] = '2014'
        df_2015 = pd.read_csv(r'..\data\raw\motivation_2015.csv', usecols = columns)
        df_2015['year'] = '2015'

        df_2014.rename(columns={"COHORT": "cohort"}, inplace=True)

        # Concatenate datasets
        df = pd.concat([df_2014, df_2015])

        ## Swap 0s and 1s in the target columns ['reenrolled', 'bsa_dummy']
        df['bsa_dummy'] = df['bsa_dummy'].replace({0:1, 1:0})
        df['reenrolled'] = df['reenrolled'].replace({0:1, 1:0})

        # drop null values in target columns
        df.dropna(subset=['reenrolled','bsa_dummy','motivation'], inplace=True)

        df.to_csv(r'..\data\processed\motivation.csv', index=False)
        print('concatenated dataset is written on "\data\processed\motivation.csv"')
        return df

    elif status == 'read':
        df = pd.read_csv(r'..\data\processed\motivation.csv')
        return df


if __name__ == "__main__":
    df = read_df()
    print(df)




