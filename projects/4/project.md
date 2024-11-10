# **Section 4: Random Forest**  
## **Assignment Title: Random Forest Classification vs. Regression**

Project: Customer Churn Prediction
* Build a random forest model to predict customer churn for a telecom company.
* Use a dataset with customer information and churn status.
* Handle missing data and address class imbalance in the dataset.

## Objective
Apply Random Forest models to both classification and regression tasks, and compare their performance to simpler models (e.g., decision trees).

## Dataset
Use the Telco Customer Churn dataset, which includes information about telecom customers and whether they left within the last month.

## Tasks
1. Data Preprocessing:
    * Load and explore the Telco Customer Churn dataset.
    * Handle missing values and perform necessary feature engineering.
    * Encode categorical variables appropriately.
    * Split the data into training and testing sets (70% training, 30% testing).
2. Random Forest Implementation:
    * Implement a random forest classifier from scratch using Python.
    * Include feature randomness in the tree-building process.
    * Implement majority voting for final predictions.
3. Handling Imbalanced Data:
    * Implement techniques to handle class imbalance (e.g., oversampling, undersampling, or SMOTE).
    * Compare the performance of your model with and without balancing techniques.
4. Model Training and Evaluation:
    * Train your random forest model on the training data.
    * Evaluate the model's performance on the test data using accuracy, precision, recall, F1-score, and ROC-AUC.
5. Feature Importance:
    * Implement a method to calculate feature importance in your random forest.
    * Visualize and analyze the most important features for predicting churn.
6. Hyperparameter Tuning:
    * Implement a simple grid search to tune key hyperparameters (e.g., number of trees, max depth).
    * Analyze the impact of different hyperparameters on model performance.
7. Comparison with Scikit-learn:
    * Implement the same task using scikit-learn's RandomForestClassifier.
    * Compare the performance and feature importance with your implementation.
8. Report:
    * Write a comprehensive report (max 2500 words) detailing your methodology, results, and analysis.
    * Discuss the effectiveness of random forests for churn prediction and the impact of handling imbalanced data.