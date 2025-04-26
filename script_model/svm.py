import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from matplotlib import pyplot as plt

# Load the datasets
df_svm = pd.read_csv('/Users/iman/Projects/seminar/voters/results/few_shot/final_df_fs.csv')


X = df_svm.iloc[:, 10:]
y = df_svm['vote']
X = X[['age', 'female', 'edu', 'emp', 'hhincome', 'east', 'religious',
    'leftright', 'party', 'inequality', 'immigration']]
X.columns


# Identify non-numeric columns and encode them
non_numeric_columns = X.select_dtypes(include=['object', 'category']).columns
X_features_encoded = X.copy()

for col in non_numeric_columns:
    le = LabelEncoder()
    X_features_encoded[col] = le.fit_transform(X_features_encoded[col].astype(str))

# Scale the features for SVM
scaler = StandardScaler()
X_features_scaled = scaler.fit_transform(X_features_encoded)

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_features_scaled, 
    y, 
    test_size=0.3, 
    random_state=76, 
    stratify=y
)

# Kernel: Change to 'linear', 'poly', or 'sigmoid' for different decision boundaries.
# Initialize and train the SVM classifier
svm_classifier = SVC(kernel='linear', random_state=76, probability=True)
svm_classifier.fit(X_train, y_train.values.ravel())

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the results
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", conf_matrix)

# Get feature importances (coefficients)
feature_importances = svm_classifier.coef_[0]

# Plot feature importances
plt.figure(figsize=(10, 10))
plt.bar(range(len(feature_importances)), feature_importances)
plt.xlabel('Feature Index')
plt.ylabel('Feature Importance')
plt.title('Feature Importances from Linear SVM')
plt.show()