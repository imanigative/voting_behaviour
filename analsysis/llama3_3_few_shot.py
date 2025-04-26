import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.metrics import classification_report

# Load the CSV file
df = pd.read_csv('/Users/iman/Projects/seminar/voters/results/few_shot/final_df_fs.csv')

# Define the true labels and predictions
true_labels = df['vote']
predictions = df['predicted_vote']

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)
print(f'Accuracy: {accuracy}')

df['vote'].value_counts()

party_columns = ['CDU/CSU', 'SPD', 'Die Grünen', 'AfD', 'Die Linke', 'FDP']

# Generate classification report
class_report = classification_report(true_labels, predictions)
print('Classification Report:')
print(class_report)

# Calculate misclassification rate
misclassification_rate = 1 - accuracy
print(f'Misclassification Rate: {misclassification_rate}')

# Calculate F1 score
f1 = f1_score(true_labels, predictions, average='weighted')
print(f'F1 Score: {f1}')


# Generate confusion matrix
conf_matrix = confusion_matrix(true_labels, predictions, labels=true_labels.unique())
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=party_columns)

# Plot confusion matrix and save as .png file
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title('')
plt.xticks(rotation=45)
plt.savefig('/Users/iman/Projects/seminar/voters/results/png/confusion_matrix_fs.png')
plt.show()




# Plot accuracy measures
metrics = ['Accuracy', 'Misclassification Rate', 'F1 Score']
values = [accuracy, misclassification_rate, f1]

plt.figure(figsize=(8, 6))
plt.bar(metrics, values, color=['blue', 'red', 'green'])
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Comparison of Accuracy Measures')
plt.show()


party_column = ['CDU/CSU', 'SPD', 'Grünen', 'AfD', 'Die Linke', 'FDP']

# Group by 'vote' column and calculate the average of 'party_column'
grouped_df = df.groupby('vote')[party_column].mean().reset_index()

# Print the grouped dataframe
print(grouped_df)

colors = ['black', 'red', 'green', 'blue', 'purple', 'yellow']
# Plot the average of 'party_column' for each 'vote' group as a bar chart
grouped_df.set_index('vote', inplace=True)
grouped_df.plot(kind='bar', figsize=(12, 8), color=colors)

plt.xlabel('actual Vote')
plt.ylabel('Average of predicted Party Probability')
plt.title('')
plt.legend(title='Party')
plt.savefig('/Users/iman/Projects/seminar/voters/results/png/bar_fs_probs.png')
plt.show()



# Filter the dataset where 'hhincome' is 'niedriges'
filtered_df = df[(df['hhincome'] == 'niedriges') & (df['female'] == 'weiblich') & (df['age'] == 74)]
filtered_df = df[(df['leftright'] == 'mittig links')]
# Calculate the average of party columns for the filtered dataset
average_party_columns = filtered_df[party_columns].mean()

# Print the average of party columns
print(average_party_columns)

#few shot
#0.6212871287128713