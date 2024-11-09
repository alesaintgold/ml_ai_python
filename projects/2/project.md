# Project: House Price Prediction

* Develop a decision tree regressor to predict house prices based on various features.
* Use a dataset like the Boston Housing dataset or Ames Housing dataset.
* Implement pruning techniques and compare the performance before and after pruning.

## Objective
Develop a decision tree regressor to predict house prices, applying concepts from Section 2: Decision Trees in Regression.

## Dataset
Use the Ames Housing dataset, which includes various features of houses in Ames, Iowa, along with their sale prices.

## Tasks
1. Data Preprocessing:
    * Load and explore the Ames Housing dataset.
    * Handle missing values and perform necessary feature engineering.
    * Split the data into training and testing sets (70% training, 30% testing).
2. Decision Tree Regressor Implementation:
    * Implement a decision tree regressor from scratch using Python.
    * Use mean squared error (MSE) as the splitting criterion.
    * Implement a maximum depth parameter to control tree growth.
3. Model Training and Evaluation:
    * Train your implemented model on the training data.
    * Evaluate the model's performance on the test data using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
4. Pruning Implementation:
    * Implement post-pruning techniques as discussed in the course.
    * Compare the performance of the pruned and unpruned trees.
5. Visualization and Analysis:
    * Visualize the decision tree before and after pruning.
    * Analyze the impact of different pruning levels on model performance.
6. Comparison with Scikit-learn:
    * Implement the same regression task using scikit-learn's DecisionTreeRegressor.
    * Compare the performance and tree structure with your implementation.
7. Report:
    * Write a comprehensive report (max 2000 words) detailing your methodology, results, and analysis.
    * Discuss the effectiveness of pruning and its impact on model performance.

## Deliverables: 
1. Python code for decision tree regression and pruning.  
2. Performance evaluation (RMSE or MAE) before and after pruning.  
3. Plots showing predicted vs actual values for both pruned and unpruned trees.  
4. A report analyzing the impact of pruning on model accuracy, overfitting, and bias-variance tradeoff.

