# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- **Model type**: Supervised classification using a Random Forest Classifier  
- **Framework**: scikit-learn  
- **Training pipeline**: Custom Python script using feature engineering, one-hot encoding, and a Random Forest model trained on tabular census data  
- **Author**: Tony Calderoni  
- **Version**: 1.0  
- **Release Date**: July 2025

## Intended Use

This model is meant for general analysis. Such as helping researchers, companies, or applications better understand income trends across different groups of people. It should not be used for decisions like hiring, lending, or insurance without careful oversight, since that could lead to unfair treatment.

## Training Data

The model was trained on the cleaned version of the [Adult Census Income dataset](https://archive.ics.uci.edu/ml/datasets/adult).  
- **Split**: 80% training / 20% test  
- **Preprocessing**: One-hot encoding of categorical variables, label binarization for the target variable `salary`

## Evaluation Data

- The test set is a stratified random split from the original dataset (20% of total records).
- Each categorical feature was also evaluated independently to assess fairness and consistency across demographic slices.

## Metrics

The overall model performance on the test set:
- **Precision**: 0.7445  
- **Recall**: 0.6599  
- **F1 Score**: 0.6997  

**Slice-level Performance Highlights**:
- `workclass=Without-pay`: F1 = 1.0000 (very small sample size of 4)
- `education=Masters`: F1 = 0.8409
- `education=7th-8th`: F1 = 0.0000
- `marital-status=Married-civ-spouse`: F1 = 0.7116
- `race=White`: F1 = 0.6850
- `sex=Female`: F1 = 0.6015, `sex=Male`: F1 = 0.6997

These slice results suggest strong performance in higher education and management roles but diminished performance among underrepresented or small demographic groups.

## Ethical Considerations

- **Bias and Fairness**: The model may not be fair for everyone. It performs worse on groups with lower education levels or from countries with fewer data points.
- **Small Sample Groups**: Some slices such as `occupation=Armed-Forces` or `native-country=Yugoslavia` showed perfect scores but were based on very few records, which can be misleading.
- **Deployment Risk**: If used to make important decisions, such as job hiring, it could unintentionally reinforce biases already present in society.
## Caveats and Recommendations

- **Sample Imbalance**: Small groups can skew results: A few examples don’t tell the whole story.
- **Update Needs**: Keep the model updated: Retrain it if the data starts to feel outdated
- **Model Interpretability**: While Random Forests offer robust performance, they are not easily interpretable. Consider SHAP values or LIME for explanations.
- **Additional Validation**: Fairness audits and additional bias-mitigation strategies are recommended before using in sensitive applications.