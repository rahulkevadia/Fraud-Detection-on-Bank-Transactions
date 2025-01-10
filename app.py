import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
df = pd.read_csv('C:/Users/Rahul/Desktop/Projects/Fraud-Detection-on-Bank-Transactions/Dataset/Transactions Dataset.csv', nrows=10000)
# Display basic information about the dataset
print(df.info())
print(df.head())


# Check for missing values
print(df.isnull().sum())
#Type conversion
label_encoder = LabelEncoder()
df.columns = df.columns.str.strip()
df['type'] = label_encoder.fit_transform(df['type'])
df = pd.get_dummies(df, columns=['type'])

non_numeric_columns = df.select_dtypes(include=['object']).columns
print("Non-numeric columns:", non_numeric_columns)
df = pd.get_dummies(df, columns=non_numeric_columns)

df['type'] = {'PAYMENT': 1, 'TRANSFER': 2, 'CASH_OUT': 3, 'CASH_IN': 4, 'DEBIT': 5}
df['type'] = df['type'].map(df['type'])



# Encode categorical variables if any (example)
# df['category'] = df['category'].astype('category').cat.codes

# Separate features and target variable
X = df.drop('isFraud', axis=1)
y = df['isFraud']
X.fillna(0, inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

from imblearn.over_sampling import SMOTE

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train the model on the resampled data
model.fit(X_train_res, y_train_res)

# Evaluate the model again
y_pred_res = model.predict(X_test)
print(f'Accuracy after SMOTE: {accuracy_score(y_test, y_pred_res)}')
print('Confusion Matrix after SMOTE:')
print(confusion_matrix(y_test, y_pred_res))
print('Classification Report after SMOTE:')
print(classification_report(y_test, y_pred_res))

# Get feature importances
importances = model.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plot feature importances
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.gca().invert_yaxis()
plt.show()

import joblib

# Save the model to a file
joblib.dump(model, 'fraud_detection_model.pkl')

# Load the model from the file
loaded_model = joblib.load('fraud_detection_model.pkl')

# Make predictions on new data
new_data = pd.read_csv('C:/Users/Rahul/Desktop/Projects/Fraud-Detection-on-Bank-Transactions/Dataset/Transactions Dataset.csv')  # Replace with your new data file
new_data_scaled = scaler.transform(new_data)
predictions = loaded_model.predict(new_data_scaled)

# Add predictions to the new dataset
new_data['isFraud'] = predictions
