import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the datasets
X_path = 'X.csv'  # Replace with actual path
y_path = 'y.csv'  # Replace with actual path

X = pd.read_csv(X_path)
y = pd.read_csv(y_path)

# Drop the ID column from X and keep it separate if needed for evaluation
X_features = X.drop(columns=X.columns[0])  # Drop the ID column (first column)

# Align indices of X and y
X_features = X_features.set_index(y.index)

# Identify non-numeric columns and encode them
non_numeric_columns = X_features.select_dtypes(include=['object', 'category']).columns
X_features_encoded = X_features.copy()

for col in non_numeric_columns:
    le = LabelEncoder()
    X_features_encoded[col] = le.fit_transform(X_features_encoded[col].astype(str))


# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_features_encoded, 
    y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y
)

# Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100)
rf_classifier.fit(X_train, y_train.values.ravel())

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the results
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", conf_matrix)

