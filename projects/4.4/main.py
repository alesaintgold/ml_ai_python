import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Load the dataset
data = pd.read_csv("creditcard.csv") 
# the file has been removed due to its size being larger than the threshold Github suggests
# If interested in replicating, the file can be found at:
# https://github.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/blob/master/creditcard.csv  

# Using a smaller sample to reduce time needed to training 
data = data.sample(frac=0.1, random_state=42)

# Separate features and target variable
X = data.drop("Class", axis=1)
y = data["Class"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check class distribution
print("Class distribution before balancing:", Counter(y_train))

# Train Random Forest classifier without balancing
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Performance without balancing:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Check class distribution after SMOTE
print("Class distribution after SMOTE:", Counter(y_train_smote))

# Train Random Forest classifier with SMOTE
rf_model_smote = RandomForestClassifier(random_state=42)
rf_model_smote.fit(X_train_smote, y_train_smote)

# Make predictions
y_pred_smote = rf_model_smote.predict(X_test)

# Evaluate performance
accuracy_smote = accuracy_score(y_test, y_pred_smote)
precision_smote = precision_score(y_test, y_pred_smote)
recall_smote = recall_score(y_test, y_pred_smote)
f1_smote = f1_score(y_test, y_pred_smote)

print("Performance with SMOTE:")
print("Accuracy:", accuracy_smote)
print("Precision:", precision_smote)
print("Recall:", recall_smote)
print("F1-score:", f1_smote)

# Apply undersampling
undersampler = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

# Check class distribution after undersampling
print("Class distribution after undersampling:", Counter(y_train_under))

# Train Random Forest classifier with undersampling
rf_model_under = RandomForestClassifier(random_state=42)
rf_model_under.fit(X_train_under, y_train_under)

# Make predictions
y_pred_under = rf_model_under.predict(X_test)

# Evaluate performance
accuracy_under = accuracy_score(y_test, y_pred_under)
precision_under = precision_score(y_test, y_pred_under)
recall_under = recall_score(y_test, y_pred_under)
f1_under = f1_score(y_test, y_pred_under)

print("Performance with Undersampling:")
print("Accuracy:", accuracy_under)
print("Precision:", precision_under)
print("Recall:", recall_under)
print("F1-score:", f1_under)

# Create a comparison table
data = {'Model': ['No Balancing', 'SMOTE', 'Undersampling'],
        'Accuracy': [accuracy, accuracy_smote, accuracy_under],
        'Precision': [precision, precision_smote, precision_under],
        'Recall': [recall, recall_smote, recall_under],
        'F1-score': [f1, f1_smote, f1_under]}
df = pd.DataFrame(data)
print(df)
