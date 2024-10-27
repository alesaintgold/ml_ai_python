# Project: Credit Card Fraud Detection

* Implement a bagging classifier to detect fraudulent credit card transactions.
* Use a publicly available credit card fraud dataset.
* Compare the performance of your bagging implementation with a single decision tree.


## Objective
Implement a bagging classifier to detect fraudulent credit card transactions, applying concepts from Section 3: Bagging.

## Dataset
Use the Credit Card Fraud Detection dataset from Kaggle, which contains transactions made by credit cards in September 2013 by European cardholders.

## Tasks
1. Data Preparation:
    * Load and explore the Credit Card Fraud Detection dataset.
    * Handle the class imbalance issue (fraudulent transactions are typically rare).
    * Split the data into training and validation sets (80% training, 20% validation).
2. Bagging Classifier Implementation:
    * Implement a bagging classifier from scratch using Python.
    * Use decision trees as base estimators.
    * Implement bootstrap sampling for creating diverse training subsets.
3. Model Training and Evaluation:
    * Train your bagging classifier on the training data.
    * Evaluate the model's performance on the validation set using precision, recall, F1-score, and ROC-AUC.
4. Out-of-Bag Error Estimation:
    * Implement Out-of-Bag (OOB) error estimation as discussed in the course.
    * Compare OOB error estimates with validation set performance.
5. Comparison with Single Decision Tree:
    * Train a single decision tree classifier on the same data.
    * Compare the performance of the bagging classifier with the single decision tree.
6. Visualization and Analysis:
    * Visualize the ROC curves for both the bagging classifier and single decision tree.
    * Analyze the impact of the number of base estimators on model performance.
7. Report:
    * Write a detailed report (max 2000 words) explaining your methodology, results, and analysis.
    * Discuss the advantages of bagging in the context of fraud detection.
