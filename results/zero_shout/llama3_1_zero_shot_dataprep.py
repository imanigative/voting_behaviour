import pandas as pd
import glob

# Define the path to the CSV files
file_paths = glob.glob('/Users/iman/Projects/seminar/voters/results/zero_shout/*.csv')

# Read and concatenate all CSV files
df_list = [pd.read_csv(file) for file in file_paths]
combined_df = pd.concat(df_list, ignore_index=True)

# Drop rows where the 'vote' column has values 'Nicht gewählt', 'Andere Partei', 'Ungültig gewählt'
combined_df = combined_df[~combined_df['vote'].isin(['Nicht gewählt', 'Andere Partei', 'Ungültig gewählt'])]


# Order the DataFrame by the 'lfdn' column in descending order
df = combined_df.sort_values(by='lfdn', ascending=True)

# Save the combined and ordered DataFrame to a new CSV file
df = df.iloc[:, :-1]

# Define the list of party columns
party_columns = ['CDU/CSU', 'SPD', 'Bündnis 90/Die Grünen', 'AfD', 'Die Linke', 'FDP']
# Remove rows where the sum of party_columns values is zero
df = df[df[party_columns].sum(axis=1) != 0]

# Function to scale the values to sum up to 100
def scale_to_100(row):
    total = row[party_columns].sum()
    if total != 100:  # Ensure total is not zero
        row[party_columns] = (row[party_columns] / total) * 100
    return row

# Apply the function to each row
scaled_df = df.apply(scale_to_100, axis=1)


scaled_df['predicted_vote'] = scaled_df[party_columns].apply(lambda row: row.idxmax(), axis=1)

scaled_df['predicted_vote'] = scaled_df['predicted_vote'].replace('Bündnis 90/Die Grünen', 'Die Grünen')
scaled_df['vote'] = scaled_df['vote'].replace('Bündnis 90/Die Grünen', 'Die Grünen')
# Save the combined DataFrame to a new CSV file
scaled_df.to_csv('/Users/iman/Projects/seminar/voters/results/zero_shout/final_df_zs.csv', index=False)