# Random Forest Classifier Performance with Class Imbalance

## Class Distribution

Before applying any balancing technique, the class distribution is highly imbalanced:
- Class 0: 22,744 samples
- Class 1: 40 samples

This severe imbalance is expected to skew the model’s ability to effectively classify the minority class (Class 1).

## Model Performance without Balancing

- **Accuracy**: 99.88%
- **Precision**: 42.86%
- **Recall**: 50.00%
- **F1-score**: 46.15%

Although the model has a high accuracy, it struggles to predict the minority class (Class 1) effectively. The low precision indicates that most of the predictions for Class 1 are false positives. The relatively low recall suggests the model is not able to identify many of the true Class 1 samples.

## Performance After SMOTE (Synthetic Minority Over-sampling Technique)

After applying SMOTE, the class distribution becomes balanced:
- Class 0: 22,744 samples
- Class 1: 22,744 samples

- **Accuracy**: 99.91%
- **Precision**: 57.14%
- **Recall**: 66.67%
- **F1-score**: 61.54%

SMOTE has a noticeable effect on recall and F1-score, especially for the minority class. By oversampling the minority class, recall increases to 66.67%, which means the model is better at identifying true positive instances of Class 1. Precision also improves, although it is still lower than the recall, indicating some trade-off between precision and recall. However, SMOTE may introduce noise due to the synthetic data, which could impact precision further.

## Performance After Undersampling

Undersampling reduces the majority class (Class 0) to match the size of the minority class (Class 1):
- Class 0: 40 samples
- Class 1: 40 samples

- **Accuracy**: 97.72%
- **Precision**: 3.03%
- **Recall**: 66.67%
- **F1-score**: 5.80%

Undersampling results in a lower overall accuracy compared to SMOTE, but with improved recall. The drastic drop in precision (3.03%) suggests that undersampling has caused the model to become highly biased toward predicting the minority class, as it no longer has sufficient majority class data to train on. The F1-score is very low, showing that undersampling is not an optimal solution for maintaining a balance between precision and recall.

## Discussion

Class imbalance has a profound impact on model performance, especially when dealing with classifiers like Random Forest, which tend to favor the majority class. The different balancing techniques, SMOTE and undersampling, each have their pros and cons:

- **SMOTE**: It performs the best in terms of increasing both recall and F1-score for the minority class, though at the cost of reduced precision. SMOTE can help the model learn better representations of the minority class, but the introduction of synthetic samples could lead to some noise, potentially harming precision.

- **Undersampling**: While it balances the class distribution by reducing the majority class, it leads to the loss of valuable information, resulting in lower accuracy and precision. Although recall is high, the overall model performance suffers due to the reduced training data for the majority class.

In conclusion, **SMOTE** is the most balanced approach for this scenario, providing the best trade-off between recall and precision, while undersampling seems to be a less effective choice due to the loss of data. The optimal technique may vary depending on the specific dataset, the problem domain, and the importance of precision versus recall.

