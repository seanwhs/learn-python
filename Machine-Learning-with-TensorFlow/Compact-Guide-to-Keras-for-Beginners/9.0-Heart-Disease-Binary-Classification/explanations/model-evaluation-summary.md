# Model Evaluation

This document explains the evaluation process for the heart disease prediction model. The evaluation involves several key metrics and methods to assess the performance of the classification model.

## 1. Prediction on Test Data

```python
y_pred = model.predict(X_test).flatten()
y_pred_binary = (y_pred > 0.5).astype(int)
```

- **`model.predict(X_test)`**: Generates predictions on the test data, returning probabilities of the positive class.
- **`flatten()`**: Converts the 2D array of predictions to a 1D array.
- **`(y_pred > 0.5).astype(int)`**: Converts probabilities to binary class predictions. A threshold of 0.5 is used to classify the instance as a heart patient (1) or healthy (0).

## 2. Confusion Matrix

```python
conf_matrix = confusion_matrix(y_test, y_pred_binary)
print("Confusion Matrix:\n", conf_matrix)
```

- **Confusion Matrix**: A table that summarizes the performance of a classification model by showing the counts of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).
  - **True Positive (TP)**: Correctly predicted heart patients.
  - **True Negative (TN)**: Correctly predicted healthy individuals.
  - **False Positive (FP)**: Incorrectly predicted heart patients (false alarm).
  - **False Negative (FN)**: Incorrectly predicted healthy individuals (missed case).

## 3. Classification Report

```python
class_report = classification_report(y_test, y_pred_binary, target_names=['Healthy', 'Heart Patients'])
print("Classification Report:\n", class_report)
```

- **Classification Report**: Provides a detailed assessment of the model’s precision, recall, F1-score, and support for each class.
  - **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
  - **Recall (Sensitivity)**: The ratio of correctly predicted positive observations to all observations in the actual class.
  - **F1-score**: The harmonic mean of precision and recall.
  - **Support**: The number of actual occurrences of each class in the dataset.

## 4. ROC Curve and AUC

```python
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
```

- **ROC Curve (Receiver Operating Characteristic Curve)**: A plot showing the trade-off between the True Positive Rate (TPR) and the False Positive Rate (FPR) at various threshold settings.
  - **True Positive Rate (TPR)**: Same as recall; the proportion of actual positives correctly identified.
  - **False Positive Rate (FPR)**: The proportion of actual negatives incorrectly classified as positives.
  
- **AUC (Area Under the Curve)**: Represents the area under the ROC curve. It provides a single value summarizing the model's performance across all thresholds. An AUC of 1 indicates a perfect model, while an AUC of 0.5 indicates random guessing.

## 5. Visualization

```python
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()
```

- **ROC Curve Plot**: Visualizes the ROC curve, plotting TPR against FPR. The AUC value is displayed in the legend to summarize the model's performance.

## Summary

The evaluation of the model includes:
1. **Generating binary class predictions** from the probability scores.
2. **Calculating and displaying the confusion matrix** to understand the number of correct and incorrect predictions.
3. **Generating a classification report** for detailed metrics like precision, recall, F1-score, and support.
4. **Plotting the ROC curve and calculating the AUC** to assess the model’s ability to distinguish between positive and negative classes across various thresholds.

These metrics and methods provide a comprehensive view of the model's performance and its ability to make accurate predictions.
