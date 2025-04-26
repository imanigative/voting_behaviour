import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# Load the combined CSV file
df_zs = pd.read_csv('/Users/iman/Projects/seminar/voters/results/zero_shout/final_df_zs.csv')

# Define the list of parties to filter
party_columns = ['CDU/CSU', 'SPD', 'Bündnis 90/Die Grünen', 'AfD', 'Die Linke', 'FDP']

# Filter the DataFrame for the specified parties
filtered_df = combined_df[combined_df['vote'].isin(party_columns)]

filtered_df["lfdn"].unique
# Calculate the accuracy score
accuracy = accuracy_score(filtered_df['vote'], filtered_df['highest_value_column'])
print(f'Accuracy: {accuracy}')

# Calculate the F1 score
f1 = f1_score(filtered_df['vote'], filtered_df['highest_value_column'], average='weighted')
print(f'F1 Score: {f1}')

# Calculate the precision score
precision = precision_score(filtered_df['vote'], filtered_df['highest_value_column'], average='weighted')
print(f'Precision: {precision}')

# Plot the confusion matrix
conf_matrix = confusion_matrix(filtered_df['vote'], filtered_df['highest_value_column'], labels=party_columns)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=party_columns)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Plot the accuracy measures
metrics = ['Accuracy', 'F1 Score', 'Precision']
values = [accuracy, f1, precision]

plt.figure(figsize=(8, 6))
plt.bar(metrics, values, color=['blue', 'green', 'red'])
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Comparison of Accuracy Measures')
plt.show()

##########################################################################

# Calculate the average probability for each category of votes
average_probabilities_predicted = filtered_df.groupby('vote')[party_columns].mean()

# Print the average probabilities
print(average_probabilities_predicted)

# Visualize the average probabilities
average_probabilities_predicted.plot(kind='bar', figsize=(10, 6))
plt.ylabel('Average Probability')
plt.title('Average Probability of Each Category of Votes')
plt.legend(title='Party')
plt.show()

##################################################
# this wont woek because the actual df does not have probabilities
# Load the CSV file
vote_df_actual = pd.read_csv('/Users/iman/Projects/seminar/voters/data_csv/GLES2017.csv')

# Define the list of parties to filter
party_columns = ['CDU/CSU', 'SPD', 'Bündnis 90/Die Grünen', 'AfD', 'Die Linke', 'FDP']

# Filter the DataFrame for the specified parties
filtered_vote_df = vote_df_actual[vote_df_actual['vote'].isin(party_columns)]

# Calculate the average probability for each category of votes
average_probabilities_actual = filtered_vote_df.groupby('vote')[party_columns].mean()

# Print the average probabilities
print(average_probabilities_actual)

# Visualize the average probabilities
average_probabilities_actual.plot(kind='bar', figsize=(10, 6))
plt.ylabel('Average Probability')
plt.title('Average Probability of Each Category of Votes (actual)')
plt.legend(title='Party')
plt.show()

######################################################################
# Combine the predicted and actual average probabilities into a single DataFrame
combined_probabilities = pd.concat([average_probabilities_predicted, average_probabilities_actual], keys=['Predicted', 'Actual'])

# Plot the combined average probabilities
colors = ['black', 'red', 'green', 'blue', 'violet', 'yellow']
fig, ax = plt.subplots(figsize=(12, 8))

# Plot predicted probabilities
average_probabilities_predicted.plot(kind='bar', ax=ax, position=0, width=0.4, color=colors, alpha=0.7, label='Predicted')

# Plot actual probabilities
average_probabilities_actual.plot(kind='bar', ax=ax, position=1, width=0.4, color=colors, alpha=0.4, label='Actual')

plt.ylabel('Average Probability')
plt.title('Comparison of Predicted and Actual Average Probability of Each Category of Votes')
plt.legend(title='Party')
plt.show()







# Define the true labels and predictions
true_labels = df_zs['vote']
predictions = df_zs['predicted_vote']

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)
print(f'Accuracy: {accuracy}')

#Accuracy:Accuracy: 0.4800747198007472

# Generate classification report
class_report = classification_report(true_labels, predictions)
print('Classification Report:')
print(class_report)

party_columns = ['CDU/CSU', 'SPD', 'Die Grünen', 'AfD', 'Die Linke', 'FDP']

# Generate confusion matrix
conf_matrix = confusion_matrix(true_labels, predictions, labels=true_labels.unique())
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=party_columns)

# Plot confusion matrix and save as .png file
fig, ax = plt.subplots(figsize=(10, 9))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title('')
plt.xticks(rotation=45)
plt.savefig('/Users/iman/Projects/seminar/voters/results/png/confusion_matrix_zs.png')
plt.show()