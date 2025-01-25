# Report on Feature Importance for Titanic Dataset

We trained a machine learning model on the Titanic dataset to assess the importance of different features in predicting survival outcomes. The features we considered include Age, Fare, Sex, Parch, Embarked, SibSp, and Pclass. The following table summarizes the importance values derived from the model (the values did not vary between runs of the script):


| Feature  | Importance |
| ---------- | ------------ |
| Age      | 0.312596   |
| Fare     | 0.284466   |
| Sex      | 0.251952   |
| Parch    | 0.050758   |
| Embarked | 0.038600   |
| SibSp    | 0.035377   |
| Pclass   | 0.026252   |

## Interpretation of Results

1. **Age (0.312596)**: The most significant feature in predicting survival is Age, with an importance score of 0.3126. This suggests that the age of passengers had a substantial effect on survival rates, potentially due to the prioritization of women and children, or other factors like the vulnerability of older individuals.
2. **Fare (0.284466)**: Fare also plays a crucial role, with a high importance score of 0.2845. The fare paid for the ticket could correlate with the passenger's socio-economic status, which likely influenced their survival chances (e.g., first-class passengers having better access to lifeboats).
3. **Sex (0.251952)**: Gender is the third most important feature, with a score of 0.2520. Historically, women were prioritized for lifeboat access, making this a highly significant predictor of survival.
4. **Parch (0.050758)**: The number of parents or children aboard (Parch) is less significant but still noteworthy. This feature could indicate that families with children or elderly passengers might have had different survival outcomes.
5. **Embarked (0.038600)**: The embarkation port (Embarked) has a relatively low importance, suggesting that the location where passengers boarded the Titanic had a smaller effect on their likelihood of survival compared to other factors like sex, age, and fare.
6. **SibSp (0.035377)**: The number of siblings or spouses aboard (SibSp) also plays a minor role in survival prediction, with a score of 0.0354. This could imply that family size had a modest impact on survival chances, though not as strongly as other features.
7. **Pclass (0.026252)**: Finally, the class of travel (Pclass) is the least important feature, with an importance score of 0.0263. Although it reflects a passenger's socio-economic status, it seems to have less predictive power compared to fare and other factors.

## Conclusion

Based on the feature importance scores, the most influential factors for predicting survival on the Titanic were **Age**, **Fare**, and **Sex**. These features likely captured the socio-economic and demographic characteristics of passengers, which heavily influenced their survival chances. In contrast, **Pclass**, **Embarked**, **SibSp**, and **Parch** played relatively minor roles in the prediction. These findings can guide further analysis and model refinement, highlighting the importance of focusing on passenger demographics and economic status when analyzing Titanic survival data.

```

```
