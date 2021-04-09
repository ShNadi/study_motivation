import pandas as pd


def read_df():
    """ Preper dataset.

    Read datasets(2014, 2015), matche the name of columns,
    swap 0 and 1 in target column concatenate,
    and make dataset ready for further analysis
    :return: prepared dataset
    :rtype: DataFrame
    """
    # Specify column names
    columns = ['studentnr_crypt', 'cohort', 'program', 'field', 'motivation',
                'prior_educ', 'previously_enrolled', 'multiple_requests',
                'age', 'gender', 'HSGPA', 'interest', 'ase', 'reenrolled',
                'bsa_dummy']
    # Read dataset_2014
    df_2014 = pd.read_csv('../data/raw/motivatoin_2014.csv')
    # Add year as a column to the dataset
    df_2014['year'] = '2014'
    # Read dataset_2015
    df_2015 = pd.read_csv('../data/raw/motivation_2015.csv', usecols=columns)
    # Add year as a column to the dataset
    df_2015['year'] = '2015'

    df_2014.rename(columns={"COHORT": "cohort"}, inplace=True)

    # Concatenate datasets
    df = pd.concat([df_2014, df_2015])

    # Swap 0s and 1s in the target columns ['reenrolled', 'bsa_dummy']
    df['bsa_dummy'] = df['bsa_dummy'].replace({0: 1, 1: 0})
    df['reenrolled'] = df['reenrolled'].replace({0: 1, 1: 0})

    # drop null values in target columns
    df.dropna(subset=['reenrolled', 'bsa_dummy', 'motivation'], inplace=True)

    df.to_csv('../data/processed/motivation.csv', index=False)
    print('concatenated dataset is written on "/data/processed/motivation.csv"')
    return df


if __name__ == "__main__":
    df = read_df()
    print(df)
