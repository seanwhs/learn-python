# Classification Report

The **classification report** is a detailed evaluation tool for classification models, providing metrics to assess the performance of the model for each class individually. 

## Code

```python
class_report = classification_report(y_test, y_pred_binary, target_names=['Healthy', 'Heart Patients'])
print("Classification Report:\n", class_report)
```

## Metrics Explained

1. **Precision**
   - **Definition**: Precision is the ratio of correctly predicted positive observations to the total predicted positives.
   - **Formula**: \[\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}\]
   - **Interpretation**: Precision measures the accuracy of positive predictions. A high precision indicates that when the model predicts a positive class, it is likely to be correct. For example, if the model predicts a patient has heart disease, high precision means this prediction is likely to be correct.

2. **Recall (Sensitivity)**
   - **Definition**: Recall is the ratio of correctly predicted positive observations to all observations in the actual class.
   - **Formula**: \[\text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}\]
   - **Interpretation**: Recall measures the model’s ability to find all relevant cases (i.e., all actual positive cases). A high recall indicates that the model can identify most of the positive cases. For example, high recall means the model successfully identifies most patients who actually have heart disease.

3. **F1-score**
   - **Definition**: The F1-score is the harmonic mean of precision and recall.
   - **Formula**: \[\text{F1-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}\]
   - **Interpretation**: The F1-score balances precision and recall, providing a single metric that considers both false positives and false negatives. It is particularly useful when dealing with imbalanced datasets where one class is more important than the other.

4. **Support**
   - **Definition**: Support refers to the number of actual occurrences of each class in the dataset.
   - **Interpretation**: Support provides context for the precision, recall, and F1-score metrics. It indicates how many instances of each class are present in the test data. For instance, if there are 50 heart patients and 100 healthy individuals in the test set, the support for 'Heart Patients' would be 50, and for 'Healthy' it would be 100.

## Example Output

The classification report might look like this:

```
Classification Report:
                  precision    recall  f1-score   support

        Healthy       0.89      0.94      0.91        100
  Heart Patients       0.87      0.78      0.82         50

    accuracy                           0.88        150
   macro avg       0.88      0.86      0.86        150
weighted avg       0.88      0.88      0.88        150
```

- **Precision, Recall, and F1-score**: Shown for each class ('Healthy' and 'Heart Patients').
- **Support**: The number of true instances for each class.
- **Accuracy**: The overall accuracy of the model.
- **Macro avg**: The average performance metric across all classes, treating all classes equally.
- **Weighted avg**: The average performance metric across all classes, weighted by the support of each class.

## Summary

The classification report provides a comprehensive view of the model's performance, especially useful in evaluating its effectiveness across different classes and understanding its strengths and weaknesses in handling imbalanced datasets or varying costs of misclassification.
