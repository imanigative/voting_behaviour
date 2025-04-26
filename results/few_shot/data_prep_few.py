import pandas as pd
import json
import glob

# Define the path to the CSV files
file_paths = glob.glob('/Users/iman/Projects/seminar/voters/results/few_shot/*.csv')

# Read and concatenate all CSV files
df_list = [pd.read_csv(file) for file in file_paths]
combined_df = pd.concat(df_list, ignore_index=True)

# Define the columns to check for the highest value
parteien = ['CDU/CSU', 'SPD', 'Bündnis 90/Die Grünen', 'AfD', 'Die Linke', 'FDP']
parteien = ["CDU/CSU", "SPD", "Grünen", "AfD", "Die Linke", "FDP"]
# Function to extract probabilities from the raw_response column
def extract_probabilities(raw_response):
    try:
        # Parse the JSON string
        data = json.loads(raw_response.replace("'", "\""))
        # Extract probabilities for each party
        probabilities = {party['name']: party['wahrscheinlichkeit'] for party in data['parteien']}
        return probabilities
    except (json.JSONDecodeError, KeyError):
        # Return NaN if parsing fails
        return {party: float('nan') for party in parteien}


# Apply the function to the raw_response column and create new columns
probabilities_df = combined_df['raw_response'].apply(extract_probabilities).apply(pd.Series)

# Merge the new columns with the original DataFrame
df = pd.concat([combined_df[['lfdn', 'prompt_A', 'raw_response']], probabilities_df], axis=1)

# Add a new column with the name of the column that has the highest value
df['predicted_vote'] = df[parteien].idxmax(axis=1)



# Save the combined DataFrame to a new CSV file

# merge the df with 
# Read the GLES2017 dataset
gles_df = pd.read_csv('/Users/iman/Projects/seminar/voters/data_csv/filtered_GLES_vote.csv')

# Merge the df with GLES2017 on the 'lfdn' column
merged_df = pd.merge(df, gles_df[['lfdn', 'vote']], on='lfdn', how='left')

# Update the original df with the merged data
merged_df.to_csv('/Users/iman/Projects/seminar/voters/results/few_shot/combined_results_few_50_200.csv', index=False)




# Save the updated DataFrame to a new CSV file
df.to_csv('/Users/iman/Projects/seminar/voters/results/few_shot/updated_results_with_predictions_lm3_3_50_100.csv', index=False)



# Define the path to the CSV files
file_paths = glob.glob('/Users/iman/Projects/seminar/voters/results/few_shot/*.csv')

# Read and concatenate all CSV files
df_list = [pd.read_csv(file) for file in file_paths]
combined_df = pd.concat(df_list, ignore_index=True)

# Order the DataFrame by the 'lfdn' column in descending order
combined_df = combined_df.sort_values(by='lfdn', ascending=True)

# Save the combined and ordered DataFrame to a new CSV file
combined_df = combined_df.iloc[:, :-2]


# Define the list of party columns
party_columns = ['CDU/CSU', 'SPD', 'Grünen', 'AfD', 'Die Linke', 'FDP']

# Function to scale the values to sum up to 100
def scale_to_100(row):
    total = row[party_columns].sum()
    if total != 100:
        row[party_columns] = (row[party_columns] / total) * 100
    return row

# Apply the function to each row
scaled_df = combined_df.apply(scale_to_100, axis=1)


scaled_df['predicted_vote'] = scaled_df[party_columns].apply(lambda row: row.idxmax(), axis=1)

scaled_df
# Read the GLES2017 dataset
gles_df = pd.read_csv('/Users/iman/Projects/seminar/voters/data_csv/GLES2017.csv')

# Merge the scaled_df with GLES2017 on the 'lfdn' column
final_df_fs = pd.merge(scaled_df, gles_df.drop(columns=['vote']), on='lfdn', how='left')

final_df_fs['predicted_vote'] = final_df_fs['predicted_vote'].replace('Grünen', 'Bündnis 90/Die Grünen')

# Save the updated DataFrame to a new CSV file
final_df_fs.to_csv('/Users/iman/Projects/seminar/voters/results/few_shot/final_df_fs.csv', index=False)

