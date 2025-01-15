# Section 4: Introduction to Random Forest

## Exercise 4.1: Bagging vs Random Forest

**Objective:**  
Compare the performance of a bagging model and a random forest model on a classification problem.

**Instructions:**
1. Load a classification dataset (e.g., the Wine dataset).
2. Train a bagging model and a random forest model on the dataset.
3. Compare their performances in terms of accuracy, precision, recall, and computational efficiency.
4. Discuss the impact of feature randomization in Random Forest on model performance.

**Deliverables:**
- Jupyter notebook or Python script with code and comments.
- Performance comparison table for both models (accuracy, precision, recall).
- Discussion of the differences in performance and impact of feature randomization.

## Exercise 4.2: Hyperparameter Tuning

**Objective:**  
Tune the hyperparameters of a Random Forest model using GridSearchCV or RandomizedSearchCV.

**Instructions:**
1. Load a classification dataset (e.g., the Diabetes dataset).
2. Train a Random Forest classifier with different hyperparameters, such as number of trees (`n_estimators`), max depth, and min samples split.
3. Use GridSearchCV or RandomizedSearchCV to find the optimal set of hyperparameters.
4. Report the best parameters and evaluate the final model on the test set.

**Deliverables:**
- Python code showing the process of hyperparameter tuning with GridSearchCV or RandomizedSearchCV.
- Table showing the best parameters found and final evaluation metrics.
- A brief discussion on the tuning process and results.

## Exercise 4.3: Feature Importance

**Objective:**  
Use Random Forest to determine the importance of different features in a dataset.

**Instructions:**
1. Load a dataset with multiple features (e.g., Titanic dataset or any other suitable dataset).
2. Train a Random Forest classifier or regressor.
3. Extract and plot the feature importances.
4. Interpret the results and discuss which features are most important for the prediction.

**Deliverables:**
- Python code for training the Random Forest model and extracting feature importances.
- A plot of feature importances.
- A brief report interpreting the results.

## Exercise 4.4: Random Forest with Class Imbalance

**Objective:**  
Implement a Random Forest classifier for a dataset with class imbalance and apply techniques to handle this imbalance.

**Instructions:**
1. Choose a classification dataset with class imbalance (e.g., Fraud Detection dataset).
2. Train a Random Forest classifier and evaluate the performance using metrics such as accuracy, precision, recall, and F1-score.
3. Apply techniques like oversampling (SMOTE), undersampling, or class weights to handle the class imbalance.
4. Compare the model's performance before and after applying the balancing techniques.

**Deliverables:**
- Python code for implementing class imbalance handling techniques.
- Comparison table of model performance before and after applying balancing techniques.
- Discussion of the impact of handling class imbalance.
