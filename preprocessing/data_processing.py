import pandas as pd

# Importing the dataset
data = pd.read_csv('final.csv')

# removing duplicates from the dataset
data.drop_duplicates(inplace=True)
