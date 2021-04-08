import pandas as pd

def read_df( status = 'read'):
    """
This function reads datasets(2014, 2015), matches the name of columns, concatenate, and make dataset ready for further
analysis
    :param status: takes two variable, 'prepare': to match the name of columns and concatenate the datasets and write it
    in '..\data\processed\motivation.csv'. 'read': to read the dataset.
    :type status: string
    :return: concatenated dataset
    :rtype: DataFrame
    """
    if status == 'prepare':
        # Specify column names
        columns = ['studentnr_crypt', 'cohort', 'program', 'field', 'motivation',
                        'prior_educ', 'previously_enrolled', 'multiple_requests', 'age',
                        'gender', 'HSGPA', 'interest', 'ase', 'reenrolled', 'bsa_dummy']
        # Read dataset_2014
        df_2014 = pd.read_csv(r'..\data\raw\motivatoin_2014.csv')
        # Add year as a column to the dataset
        df_2014 ['year'] = '2014'
        # Read dataset_2015
        df_2015 = pd.read_csv(r'..\data\raw\motivation_2015.csv', usecols = columns)
        # Add year as a column to the dataset
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




