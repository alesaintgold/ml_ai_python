import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#loading the dataset
data = pd.read_csv("titanic.csv") 

#precomputing the data
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1}) 
data['Embarked'] = data['Embarked'].fillna('S') 
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}) 
data = data.dropna() 

X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

#determining features we want to examine and target
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
target = "Survived"

#defining and training 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#calculating, printing and plotting features importance
feature_importances = rf_model.feature_importances_

feature_importance_data = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
feature_importance_data = feature_importance_data.sort_values(by='Importance', ascending=False)

print(feature_importance_data) 

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_data['Feature'], feature_importance_data['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importances')
plt.show()
